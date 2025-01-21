import sys
import scipy
import cv2 as cv
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image
import skimage as ski
import pytesseract
import pprint


def get_rectangular_contours(contours):
    """Approximates provided contours and returns only those which have 4 vertices"""
    res = []
    for contour in contours:
        hull = cv.convexHull(contour)
        peri = cv.arcLength(hull, closed=True)
        approx = cv.approxPolyDP(hull, 0.04 * peri, closed=True)
        if len(approx) == 4:
            res.append(approx)
    return res

def py_blockproc(A, blockdims, func=0):
    vr, hr = A.shape[0] // blockdims[0], A.shape[1] // blockdims[1]
    B = np.zeros((vr,hr))
    print(B.shape)
    verts = np.vsplit(A, vr)
    for i in range(len(verts)):
       for j, v in enumerate(np.hsplit(verts[i], hr)):
          B[i,j]=(np.std(A[
             i * blockdims[0] : (i + 1) * blockdims[0],
             j * blockdims[1] : (j + 1) * blockdims[1]
            ]))
    return B

def display_segments(name, item, axis='off'):
    plt.figure(figsize=(12, 9))
    plt.imshow(item, cmap="magma")
    plt.title(name)
    plt.axis(axis)
    plt.subplots_adjust(wspace=.05, left=.01, bottom=.01, right=.99, top=.9)
    plt.show()

def get_values_from_img(roi):
     '''
      get the values of coord x and y for the image that contain the signal
      INPUT:
            roi : binary image with signal in white
      OUTPUT:
            xs, ys: values of the signal
     '''
     length = roi.shape[1]
     width =  roi.shape[0]
     xs = []
     ys = []
     for j in range(length):
         for k in range(width - 1):
             if roi[k][j] == 255:
                xs.append(j)
                ys.append(width - k)
                break
             else:
                continue
     return width, length,xs,ys

def measure_extract_pulse(x,y, verbose=0):

    min_pulse = np.min(y)
    max_pulse = np.max(y)

    height = np.max(max_pulse-min_pulse)
    threshold = height / 2
    index = np.where((y - min_pulse)>=threshold)[0]
    width = x[index[-1]] - x[index[0]]
    if verbose > 0:
        print(f"pulse height: {height}")
        print(f"pulse width: {width} time units")
    return width, height

def detect_ref_pulse(roi, template, threshold=0.6, verbose=2):
    ''' 

    '''
   
    w, h = template.shape[::-1]
    if roi.shape[0] <= template.shape[0] or roi.shape[1]<= template.shape[1]:
               
           #template is bigger then roi. Can not perform matchTemplate

            empty_list=[]
            empty_array = np.array(empty_list)
            loc = (empty_array,empty_array )
    else:
            res = cv.matchTemplate(roi,template,cv.TM_CCORR_NORMED) # try tofind the pulse using a template match
            # Getting the max
            x, y = np.unravel_index(np.argmax(res), res.shape)
            print("INFO: max correlation is {} in x = {} and y = {}.".format(np.max(res),x,y))
            plt.imshow(roi)
            template_width, template_height = template.shape
            if verbose > 1:
                plt.imshow(roi)
                rect = plt.Rectangle((y, x), template_height, template_width, color='red', 
                     fc='none')
                plt.gca().add_patch(rect)
                plt.title('Grayscale Image with Bounding Box around the pulse')
                plt.axis('off')
                plt.show()
            
            loc = np.where( res >= threshold)
           
    if len(loc[0])>0:
        detected = True # pulse was detected 

        ppts=np.array(list(map(list, zip(*loc[::-1])))) #obtain um array from the list of tuples
        print(ppts)
        ppts_max= ppts[:,0].max()
        ppts_min= ppts[:,0].min()
        ppts_median = np.median(ppts[:,0])
        print(ppts_max, ppts_median, ppts_min)
        
        # extracted_pulse = roi[:,ppts_max:ppts_max+w]
        # baseline_pulse = np.argmax(np.std(extracted_pulse, axis =1))
        # _,_,xpulse,ypulse= get_values_from_img(extracted_pulse)
        # wpulse,hpulse = measure_extract_pulse(xpulse,ypulse)
        
        extracted_pulse = roi[x:x+template_width,y:y+template_height]
        plt.imshow(extracted_pulse)
        plt.show()
        _,_,xpulse,ypulse= get_values_from_img(extracted_pulse)
        wpulse,hpulse = measure_extract_pulse(xpulse,ypulse)
       
             
        if (y > roi.shape[1]//2): 
                     #pulse detected on the right
                     print("INFO: Pulse detected on the right")
                     #roi = roi[:,:-ppts_min]
                     #TODO: return just the position of the pulse
        else:
                    #pulse detected on the left
                    print("INFO: Pulse detected on the left")
                    #roi = roi[:,ppts_max+w+1:]
                    #TODO retuern just position of the pulse
              
              
              
    else:
              # No pulse detected or the roi has no pulse
              detected = False
              #curve_scales.append((np.nan,np.nan))
              wpulse = np.nan
              hpulse = np.nan
    
    return detected,x,y, wpulse, hpulse, 

def process_line(line_number, labeled_line,offset,line_leads,config_dict, verbose = 0) :
    ''' 
    
    '''
    # TODO: Clean this dictonary
    line_dict ={}
    line_dict['width']=[]
    line_dict['length']=[]
    line_dict['lb']=[]
    line_dict['ub']= []
    line_dict['baseline'] = []
    line_dict['wpulse']=config_dict['wpulse']
    line_dict['hpulse']= config_dict['hpulse']
    line_dict['curves']=[]
    line_dict['offset_line'] = offset

    if verbose > 1: 
         display_segments("Labeled Line", labeled_line)

    u,c = np.unique(labeled_line, return_counts=True) 
    segment_labels = np.argsort(-c[1:])+1  # sort label by segment size in decresent order
    segment_length = -np.sort(-c[1:])
    max_label = np.max(u)

    app_seg_size = labeled_line.shape[1]//config_dict['layout'][1] 
    if verbose > 1:
            print("INFO: unique label {}.".format(u))
            print("INFO: count {}.".format(c))
            print("INFO: segment labels {}.".format(segment_labels))
            print("INFO: segment lenghth {}.".format(segment_length))
            
    larger_segments = segment_labels[:config_dict['layout'][1]+1]

    temp = np.round(segment_length[larger_segments]/app_seg_size,1)

    print("INFO: segment ratio {}.".format(temp))
    
   
    segment_ratios = []
    
    smallest_ratio = np.inf
    for label in larger_segments:
        sl = ndimage.find_objects(labeled_line==label)
        roi = labeled_line[sl[0][0],sl[0][1]] # slice in x and slice in y

        length = roi.shape[1]
        roi_copy = roi.copy()
        roi_copy = np.where(roi_copy==label,255,0)
        roi_copy = roi_copy.astype("uint8")

        # calculate the ratio between length and approximate segment size
        # to check if teh segmentation concatenate 2 ou more segments
        ratio = round((length/app_seg_size),1) # calculate the ratio between length and appromate segment
        

        if verbose > 1:
            print("INFO: label {}.".format(label))
            print("INFO: length {}.".format(roi_copy.shape[1]))
            print("INFO: Ratio {}.".format(ratio))
            plt.imshow(roi_copy)
            plt.show()

        ratio = round((length/app_seg_size),1)
        segment_ratios.append(ratio)

        # Based on the ratio, classifiy the segments
        #if ratio > 1 then split the segments 

        if ratio == 4.0:
            print("INFO: Four Segments {}.".format(ratio))
            print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

            if line_number == config_dict['layout'][0]+1:
                # if this line a rhythm line the the type is rhythm and just copy to  curves
                print("INFO: Line number {} and layout lines {} .".format(line_number,config_dict[0]))
                print("INFO: One Segment {}.".format(ratio))
                print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

                slx_seg1_start = offset[0] #+ sl[0][0].start
                slx_seg1_stop =  offset[1] # + sl[0][0].stop
                sly_seg1_start =  sl[0][1].start
                sly_seg1_stop =   sl[0][1].stop

                # append segment to the segment list
                segment_dict = {}
                segment_dict['line'] = line_number
                segment_dict['label'] =label
                segment_dict ['start_x'] = slx_seg1_start
                segment_dict ['stop_x'] = slx_seg1_stop
                segment_dict ['start_y'] = sly_seg1_start
                segment_dict ['stop_y'] = sly_seg1_stop

                line_dict['curves'].append(segment_dict)

            else:
                print("INFO: Four Segments")

        elif ratio == 3.0:
            print("INFO: Three Segments {}.".format(ratio))
            print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

            # Separate the segments 

             # First segment
            slx_seg1_start =offset[0] #+ sl[0][0].start
            slx_seg1_stop = offset[1] #+ sl[0][0].stop
            sly_seg1_start =  sl[0][1].start
            sly_seg1_stop =   sl[0][1].start + ( (sl[0][1].stop -sl[0][1].start)//3) 

            # append segment to the segment list
            segment_dict = {}
            segment_dict['line'] = line_number
            segment_dict['label'] =label
            segment_dict ['start_x'] = slx_seg1_start
            segment_dict ['stop_x'] = slx_seg1_stop
            segment_dict ['start_y'] = sly_seg1_start
            segment_dict ['stop_y'] = sly_seg1_stop

            line_dict['curves'].append(segment_dict)

            # Second Segment
 

            slx_seg2_start =offset[0] #+ sl[0][0].start
            slx_seg2_stop = offset[1] #+ sl[0][0].stop 
            sly_seg2_start = sl[0][1].start + ( (sl[0][1].stop -sl[0][1].start)//3) 
            sly_seg2_stop = sl[0][1].start + ( (sl[0][1].stop -sl[0][1].start)//3) + ( (sl[0][1].stop -sl[0][1].start)//3)
            
            max_label = max_label+1 # add a new label 

             # append segment to the segment list
            segment_dict = {}
            segment_dict['line'] = line_number
            segment_dict['label'] = max_label
            segment_dict ['start_x'] = slx_seg2_start
            segment_dict ['stop_x'] = slx_seg2_stop
            segment_dict ['start_y'] = sly_seg2_start
            segment_dict ['stop_y'] = sly_seg2_stop

            line_dict['curves'].append(segment_dict)

            max_label = max_label + 1

            # Third Segment
            slx_seg3_start =offset[0] + sl[0][0].start
            slx_seg3_stop = offset[1] + sl[0][0].stop 
            sly_seg3_start = sl[0][1].start + ( (sl[0][1].stop -sl[0][1].start)//3) + ( (sl[0][1].stop -sl[0][1].start)//3)
            sly_seg3_stop = sl[0][1].stop

            max_label = max_label   #add new label 
             # append segment to the segment list
            segment_dict = {}
            segment_dict['line'] = line_number
            segment_dict['label'] = max_label
            segment_dict ['start_x'] = slx_seg3_start
            segment_dict ['stop_x'] = slx_seg3_stop
            segment_dict ['start_y'] = sly_seg3_start
            segment_dict ['stop_y'] = sly_seg3_stop

            line_dict['curves'].append(segment_dict)


        elif ratio == 2.0:
            print("INFO: two Segments {}.".format(ratio))
            print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

            # Separate the segments 

            # First segment
            slx_seg1_start =offset[0] #+ sl[0][0].start
            slx_seg1_stop = offset[1] #+ sl[0][0].stop
            sly_seg1_start =  sl[0][1].start
            sly_seg1_stop =   sl[0][1].start + ( (sl[0][1].stop -sl[0][1].start)//2) 

            # append segment to the segment list
            segment_dict = {}
            segment_dict['line'] = line_number
            segment_dict['label'] =label
            segment_dict ['start_x'] = slx_seg1_start
            segment_dict ['stop_x'] = slx_seg1_stop
            segment_dict ['start_y'] = sly_seg1_start
            segment_dict ['stop_y'] = sly_seg1_stop

            line_dict['curves'].append(segment_dict)

            # Second segments 

            slx_seg2_start =offset[0] #+ sl[0][0].start
            slx_seg2_stop = offset[1] #+ sl[0][0].stop 
            sly_seg2_start = sl[0][1].start + ( (sl[0][1].stop -sl[0][1].start)//2) 
            sly_seg2_stop = sl[0][1].stop

            max_label = max_label + 1

             # append segment to the segment list
            segment_dict = {}
            segment_dict['line'] = line_number
            segment_dict['label'] =max_label
            segment_dict ['start_x'] = slx_seg2_start
            segment_dict ['stop_x'] = slx_seg2_stop
            segment_dict ['start_y'] = sly_seg2_start
            segment_dict ['stop_y'] = sly_seg2_stop

            line_dict['curves'].append(segment_dict)

        elif ratio == 1.0:
            print("INFO: One Segment {}.".format(ratio))
            print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

            slx_seg1_start = offset[0] #+ sl[0][0].start
            slx_seg1_stop =  offset[1] # + sl[0][0].stop
            sly_seg1_start =  sl[0][1].start
            sly_seg1_stop =   sl[0][1].stop

            # append segment to the segment list
            segment_dict = {}
            segment_dict['line'] = line_number
            segment_dict['label'] =label
            segment_dict ['start_x'] = slx_seg1_start
            segment_dict ['stop_x'] = slx_seg1_stop
            segment_dict ['start_y'] = sly_seg1_start
            segment_dict ['stop_y'] = sly_seg1_stop

            line_dict['curves'].append(segment_dict)

        elif ratio < 1.0:
            if (smallest_ratio) < 1:
                print("INFO: Garbage {}.".format(ratio))
                break
            else:
                if (config_dict['pulse'] == -1) or (line_number in config_dict['pulse']) :   # Check if the pulse is present
                     print("INFO: Pulse {}.".format(ratio))
                     _,_,xpulse,ypulse= get_values_from_img(roi_copy)
                     wpulse,hpulse = measure_extract_pulse(xpulse,ypulse)
                
                    #Check the pulse found

                     if wpulse-10 <= config_dict['wpulse'] and wpulse+10 >=config_dict['wpulse']:
                       print("INFO: pulse width checked {} and {}".format(wpulse, config_dict['wpulse']))

                     if hpulse-10 <= config_dict['hpulse'] and hpulse+10 >=config_dict['hpulse']:
                       print("INFO: pulse height checked {} and {}".format(wpulse, config_dict['hpulse']))
                    
                
                     line_dict ['scale']=(wpulse,hpulse)
                else:
                     #if it is no suppose to have a pulse it is garbage
                     line_dict ['scale']=(np.nan, np.nan)
                     break
               

        if ratio < smallest_ratio:
            smallest_ratio = ratio
    # Print dictionary with pprint
    pprint.pprint(line_dict)
    line_dict['curves'] = sorted(line_dict['curves'], key=lambda d: d['start_y'])

    
    for i, d in enumerate(line_dict['curves']):
        d['name'] = line_leads[i]  #add  the name of the segments

    
    pprint.pprint(line_dict)   
    return line_dict

   


                 
