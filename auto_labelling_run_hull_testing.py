from __future__ import print_function
from __future__ import division
import glob
import re
import cv2 
from math import pi
import numpy as np
import csv
import argparse
from psd_tools import PSDImage
from PIL import Image
import psapi
from utils import * 


def parse_args():
    parser = argparse.ArgumentParser(description='Auto Labeling Algo Ver 1.0')
    parser.add_argument('--brand-name', help='specify the brand name to choose the right prod contour')
    parser.add_argument('--data-dir', help='the dir of all input images')
    parser.add_argument('--label-dir', help='the dir of label images')
    parser.add_argument('--work-dir', help='the dir to save psd/png files')
    args = parser.parse_args()
    return args


def layer_blend(prod_img, 
                prod_ref,
                label_img,
                prod_pic,
                prod_name,
                product_width,
                product_height,
                label_width,
                label_height,
                left_margin,
                top_margin,
                blend_mode=psapi.enum.BlendMode.normal, # <class 'psapi.enum.BlendMode'>?
                opacity=255,
                save_path='./output/psd_name.psd',
                save_psd=False):
    '''Import imgs from tmp_dir as artLayer and write to psd file
    '''

    if prod_img.shape[-1] == 4:
        # transparent to black
        prod_img[np.where(prod_img[:, :, 3] == 0)] = (0, 0, 0, 255)
        fill_img = prod_img[:, :, 0:3]
    else:
        fill_img = prod_img
    fill_img[np.where(fill_img[:, :, 0] < 60)] = (0, 0, 0)
    fill_img[np.where(fill_img[:, :, 1] < 60)] = (0, 0, 0)
    fill_img[np.where(fill_img[:, :, 2] < 60)] = (0, 0, 0)

    prod_height, prod_width, _ = prod_img.shape # real input prod_img size
    cv2.imshow(prod_name, prod_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## extract contours of interest step 1
    fill_img = fill_img.astype(np.uint8)
    fill_img = cv2.GaussianBlur(fill_img, (15, 15), 0)
    lower, upper = get_edge_detection_thresholds(fill_img)
    edged = cv2.Canny(fill_img, 10, upper) # slightly decreased low_threshold
    gray_img = cv2.cvtColor(fill_img, cv2.COLOR_BGR2GRAY)
    _, bw_img = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Prada needs canny edge
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    
    # matchshape(predefined prod hed contour img, cropped prod img)
    ref_cnts = {}
    for pred_cnt in prod_ref:
        cnt_basename = os.path.basename(pred_cnt).split('.')[0]
        ref_cnts[cnt_basename] = np.load(pred_cnt)
    matching_score = {}
    ms_contours = []
    maskImg = np.zeros(prod_img.shape[:3], dtype=np.uint8)
    for type, ref_cnt in ref_cnts.items():
        matching_score[type] = []
        pred_bound_rect = cv2.boundingRect(ref_cnt)
        resize_shape = (int(pred_bound_rect[2]), int(pred_bound_rect[3]))
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 1e4: # (given) prod_size (1024, 1024)
                matching_score[type].append(100.0) # just any large number 
            else:
                tmp_maskImg = np.ones((prod_img.shape[0], prod_img.shape[1], 3), dtype=np.uint8)
                ref_cnt_rect = cv2.resize(pred_bound_rect, resize_shape)
                cnt_rect = cv2.boundingRect(contours[i])
                cropped_cnt_rect = bw_img[cnt_rect[1]: cnt_rect[1]+cnt_rect[3], cnt_rect[0]: cnt_rect[0]+cnt_rect[2]]
                
                cropped_cnt_rect = cv2.resize(cropped_cnt_rect, resize_shape)
                ret =cv2.matchShapes(ref_cnt, cropped_cnt_rect, 1, 0.0)
                matching_score[type].append(ret)

        sim_idx = np.argmin(matching_score[type])
        print(sim_idx, matching_score[type][sim_idx])
        ms_contours.append(contours[sim_idx])
        # smooth contour
        peri = cv2.arcLength(contours[sim_idx], True)
        best_contour = contours[sim_idx] #cv2.approxPolyDP(contours[sim_idx], 0.001 * peri, True)
        cv2.drawContours(maskImg, [best_contour], -1, (255, 255, 255), -1)
        

    cv2.imshow(prod_name+" [matchshape mask]", maskImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #max_contour = max(contours, key=lambda item: cv2.contourArea(item))
    #cv2.drawContours(prod_img, [max_contour], -1, (128, 0, 0), cv2.FILLED)
    #cv2.imshow(prod_name+" [max countour]", fill_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    if cv2.contourArea(best_contour) < 1e4:
        print("NO CONTOURS DETECTED!")
        file_dest = psapi.LayeredFile_8bit(color_mode=psapi.enum.ColorMode.rgb, 
                                       width=prod_width, 
                                       height=prod_height)
        label_data = {}
        relative_scale = min(512 / label_width, 512 / label_height) 
        res_label_img = cv2.cvtColor(cv2.resize(label_img, None, fx=relative_scale, fy=relative_scale, interpolation = cv2.INTER_LANCZOS4), cv2.COLOR_RGB2RGBA)
        rota_label_img = img_rotate(res_label_img, 0)
        rota_label_height, rota_label_width, _ = rota_label_img.shape
        label_data[-1] = rota_label_img[:, :, 3].astype(np.uint8) # A
        label_data[0] = rota_label_img[:, :, 2].astype(np.uint8) ## R
        label_data[1] = rota_label_img[:, :, 1].astype(np.uint8) ## G
        label_data[2] = rota_label_img[:, :, 0].astype(np.uint8) ## B

        label_layer = psapi.ImageLayer_8bit(label_data, 
                                        layer_name="Layer label", 
                                        width=rota_label_width, 
                                        height=rota_label_height,
                                        blend_mode=blend_mode,
                                        opacity=opacity,
                                        pos_x=int(0),
                                        pos_y=int(0)
                                        )
        prod_pic_data = {}
        prod_pic_data[0] = prod_pic[:, :, 2].astype(np.uint8) # R
        prod_pic_data[1] = prod_pic[:, :, 1].astype(np.uint8) # G
        prod_pic_data[2] = prod_pic[:, :, 0].astype(np.uint8) # B
        prod_layer = psapi.ImageLayer_8bit(prod_pic_data, 
                                       layer_name="Layer product pic", 
                                       width=prod_pic.shape[0], 
                                       height=prod_pic.shape[1],
                                       blend_mode=blend_mode,
                                       opacity=opacity,
                                       )
        # add layers and write out to psd file
        file_dest.add_layer(label_layer)
        file_dest.add_layer(prod_layer)
        file_dest.write(save_path + prod_name + '.psd')

        return
    

    eign_angle, contour_cntr, axis = get_orient(best_contour, prod_img)
    
    bound_cntr, bound_rect, bound_angle = cv2.minAreaRect(best_contour)
    #bound_width, bound_height = minrect_helper(bound_rect, bound_angle)
    if product_height > product_width:
        bound_height = max(bound_rect)
        bound_width = min(bound_rect)
    else:
        bound_height = min(bound_rect)
        bound_width = max(bound_rect)
    

    # TODO cal fit_scale algo1.1
    #fit_scale = min([(bound_width-new_left_margin)/label_img.shape[1], (bound_height-new_top_margin)/label_img.shape[0]])
    # prod_spec:prod_hed scale
    hed_spec_scale = (bound_height / product_height, bound_width / product_width)
    hed_pic_scale = min(prod_pic.shape[0] / prod_img.shape[0], prod_pic.shape[1] / prod_img.shape[1])
    spec_scale = min(label_height / product_height, label_width / product_width) 
    #print("scale_ratio", bound_height , product_height , bound_width , product_width, hed_spec_scale) 
    
    # TODO size of product in prod_pic_img to be labelled 
    real_bound_height = (bound_height - top_margin*hed_spec_scale[0]) * (prod_pic.shape[0] / prod_img.shape[0])
    real_bound_width = (bound_width - 2*left_margin*hed_spec_scale[1]) * (prod_pic.shape[1] / prod_img.shape[1]) 
    #print("boundary", real_bound_height , label_height, real_bound_height / label_height, real_bound_width / label_width, real_bound_width , label_width)
    
    relative_scale = min(real_bound_height / label_height, real_bound_width / label_width)
    res_label_img = cv2.cvtColor(cv2.resize(label_img, None, fx=relative_scale, fy=relative_scale, interpolation = cv2.INTER_LANCZOS4), cv2.COLOR_RGB2RGBA)
    rota_label_img = img_rotate(res_label_img, eign_angle)
    rota_label_height, rota_label_width, _ = rota_label_img.shape 

    # left-top offset added here
    default_left_margin = (product_width - label_width) / 2 * (bound_width / product_width)
    default_top_margin = (product_height - label_height) / 2 * (bound_height / product_height)
    new_left_margin = left_margin * (bound_width / product_width)
    new_top_margin = top_margin * (bound_height / product_height)
    print("margin", default_left_margin, default_top_margin, '*', new_left_margin, new_top_margin)
    # [-45, 45] & (-90, 45) & (45, 90) ?
    rota_offset = ((new_left_margin - default_left_margin) * np.sin(np.deg2rad(eign_angle)) + (new_top_margin - default_top_margin) * np.sin(np.deg2rad(eign_angle)), 
                   (new_left_margin - default_left_margin) * np.cos(np.deg2rad(eign_angle)) + (new_top_margin - default_top_margin) * np.cos(np.deg2rad(eign_angle)))
    offset_label = (contour_cntr[0] - prod_width // 2 + rota_offset[0], 
                    contour_cntr[1] - prod_height // 2 + rota_offset[1])
    

    print("rota label img shape", rota_label_img.shape, prod_height // 2, prod_width // 2, contour_cntr, bound_cntr)
    

    if save_psd:
        
        contour_mask = np.ones((prod_height, prod_width, 4), dtype=np.uint8) * 255 # 4-th ch to represent Alpha channel
        cv2.drawContours(contour_mask, [best_contour], -1, (0, 0, 0), -1)
        cv2.drawContours(contour_mask, [np.int64(cv2.boxPoints(cv2.minAreaRect(best_contour)))], 0, (0, 0, 128), 2)
        contour_data = {}
        contour_data[-1] = contour_mask[:, :, 3].astype(np.uint8)   # A
        contour_data[0] = contour_mask[:, :, 2].astype(np.uint8)    # R
        contour_data[1] = contour_mask[:, :, 1].astype(np.uint8)    # G
        contour_data[2] = contour_mask[:, :, 0].astype(np.uint8)    # B

        contour_layer = psapi.ImageLayer_8bit(
            contour_data,
            layer_name="Layer prod contour",
            width=prod_width,
            height=prod_height,
            blend_mode=blend_mode,
            opacity=opacity,
            )

        file_dest = psapi.LayeredFile_8bit(color_mode=psapi.enum.ColorMode.rgb, 
                                       width=prod_width, 
                                       height=prod_height)
        label_data = {}
        label_data[-1] = rota_label_img[:, :, 3].astype(np.uint8) # A
        label_data[0] = rota_label_img[:, :, 2].astype(np.uint8) ## R
        label_data[1] = rota_label_img[:, :, 1].astype(np.uint8) ## G
        label_data[2] = rota_label_img[:, :, 0].astype(np.uint8) ## B

        label_layer = psapi.ImageLayer_8bit(label_data, 
                                        layer_name="Layer label", 
                                        width=rota_label_width, 
                                        height=rota_label_height,
                                        blend_mode=blend_mode,
                                        opacity=opacity,
                                        pos_x=int(offset_label[0]),
                                        pos_y=int(offset_label[1])
                                        )

        prod_hed_data = {}
        prod_hed_data[0] = prod_img[:, :, 2].astype(np.uint8) # R
        prod_hed_data[1] = prod_img[:, :, 1].astype(np.uint8) # G
        prod_hed_data[2] = prod_img[:, :, 0].astype(np.uint8) # B
        prod_hed_layer = psapi.ImageLayer_8bit(prod_hed_data, 
                                       layer_name="Layer product hed", 
                                       width=prod_width, 
                                       height=prod_height,
                                       blend_mode=blend_mode,
                                       opacity=opacity,
                                        )
        prod_pic_data = {}
        prod_pic_data[0] = prod_pic[:, :, 2].astype(np.uint8) # R
        prod_pic_data[1] = prod_pic[:, :, 1].astype(np.uint8) # G
        prod_pic_data[2] = prod_pic[:, :, 0].astype(np.uint8) # B
        prod_layer = psapi.ImageLayer_8bit(prod_pic_data, 
                                       layer_name="Layer product pic", 
                                       width=prod_pic.shape[0], 
                                       height=prod_pic.shape[1],
                                       blend_mode=blend_mode,
                                       opacity=opacity,
                                       )
        # add layers and write out to psd file
        file_dest.add_layer(label_layer)
        file_dest.add_layer(prod_hed_layer)
        file_dest.add_layer(contour_layer)
        file_dest.add_layer(prod_layer)
        file_dest.write(save_path + prod_name + '.psd')

        # add save to png (RGBA -> BGRA -> transpose -> save)
        merged_data = PSDImage.open(save_path + prod_name + '.psd') # PSDImage
        merged_image = merged_data.composite(force=True, color=1.0, alpha=0.0) # PILImage 
        split_r, split_g, split_b, split_a = np.array(merged_image).T
        merged_image = np.array([split_b, split_g, split_r, split_a])
        merged_image = Image.fromarray(merged_image.transpose())
        merged_image.save(save_path + prod_name + '_blend.png')
        
    else:
        # save blend as png sbs [prod_img | blend(prod_contour, scaled_label)
        # TODO
        #cv2.imwrite("./res_rota_15.png", rota_label_img)
        pass
    

def run_test(args, prod_spec):
    
    save_dir_mode = {'psd': os.path.join(args.work_dir, './output/'), 
                     'png': os.path.join(args.work_dir, './data/output/auto_labeling/')}
    imgs_path = glob.glob(args.data_dir + "*hed*.png") + glob.glob(args.data_dir + "*HED*.png")
    print("imgs_path", imgs_path)
    
    imgs_path_by_brand = []
    for img_path in imgs_path:
        if re.search(args.brand_name, img_path, re.IGNORECASE):
            imgs_path_by_brand.append(img_path)

    labels_path = glob.glob(args.label_dir+"*.png")
    label_path_by_brand = []
    for label_path in labels_path:
        if re.search(args.brand_name, label_path, re.IGNORECASE):
            label_path_by_brand.append(label_path)
    if len(label_path_by_brand) == 0:
        label_path_by_brand = ["./data/tmp/tmp_label.png"]
    print("label path", label_path)

    label_img = cv2.imread(label_path_by_brand[0], -1)
    prod_hed_ref = glob.glob(args.data_dir+"/"+args.brand_name+"*.npy")

    for img_path in imgs_path_by_brand:
        prod_name = os.path.basename(img_path).split('.')[0].replace('-hed', '').replace('_HED', '')
        print("prod name", prod_name)
        prod_hed_img = cv2.imread(img_path, -1) # 4ch
        prod_img = cv2.imread(os.path.join(os.path.dirname(img_path), prod_name)+'.png')
        #if "rota_0" not in prod_name: # weired case
        #    continue
        layer_blend(prod_hed_img,   # (hed w bg)
                    prod_hed_ref,   # (hed wo bg)
                    label_img,      # (brand logo rgba)
                    prod_img,       # (prod img)
                    prod_spec['product_name']+'#'+prod_name,  # (given)
                    product_width=float(prod_spec['product_width']),  # (given) set layer width
                    product_height=float(prod_spec['product_length']), # (given) set layer height
                    label_width=float(prod_spec['label_width']), # (given) 
                    label_height=float(prod_spec['label_length']), # (given)
                    left_margin=float(prod_spec['left_margin_width']), # (given)
                    top_margin=float(prod_spec['top_margin_width']),  # (given)
                    #blend_mode=lr_src.blend_mode, 
                    save_path=save_dir_mode['psd'],
                    save_psd=True)


if __name__ == "__main__":
    ### IMPORTANT NOTE: This VER only work with weak background overlay. ### 
    args = parse_args()
    brand = args.brand_name    # ['UNBRAND']
    
    angle_list = [-85, -75, -65, -45, -15, 0, 15, 45, 65, 75, 85]
    prod_spec_temp = {'brand_name': 'unbrand', 
                      'product_name': 'unnamed', 
                      'product_width': 1024, 
                      'product_length': 1024, 
                      'label_width': 1024, 
                      'label_length': 1024, 
                      'left_margin_width': 5, 
                      'top_margin_width': 5, 
                      'label_id': 'NA', 
                      'label_name': 'unnamed'}
    #create_rota(args.data_dir+"LOR001-HR-12-redandgold-background-3", angle_list, args.data_dir)
    #print(args.data_dir)

    # read csv to dict
    csvfile = open(os.path.join(args.label_dir, 'data_autolable_algo_test.csv'), newline='')
    csv_reader = csv.DictReader(csvfile)
        #for row in csv_reader:
        #    print(row.keys())
    
    if brand == 'UNBRAND':
        prod_spec = [prod_spec_temp]
    else:
        prod_spec = [row for row in csv_reader if row['brand_name'].upper().split(' ')[0] in brand]
        
    run_test(args, prod_spec[0])

