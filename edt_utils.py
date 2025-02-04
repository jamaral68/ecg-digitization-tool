import sys
import scipy
import cv2 as cv
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import interpolate
from matplotlib import pyplot as plt
from PIL import Image
import skimage as ski
import pytesseract
import pprint
import math
from itertools import groupby
import ss

def is_nan(value):
    try:
        return math.isnan(float(value))
    except ValueError:
        return False
    
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
         for k in range(width-1, 0, -1): # try of fix the letter in the signal
             if roi[k][j] == 255:
                xs.append(j)
                ys.append(width- k)
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

def convert_to_secmv(xs,ys,wp, hp,ws, baseline, pulse_per_sec,pulse_per_mv):
     '''
     INPUTS:
            xs: x-axis in pts
            ys: y-axis in pts
            wp: pulse width in pts
            hp: pulse height in pts
            baseline: segment baseline in pts
            ws:  segment width in pts 
    
     '''
     zero_line = ws -baseline
     ymv = (ys - zero_line)/(hp*pulse_per_mv) 

     sec_per_pts = (pulse_per_sec/wp)

     xsec = sec_per_pts*np.array(xs)


     return xsec, ymv



def detect_ref_pulse(roi, template,location = 'right', threshold=0.6, verbose=2):
    ''' 

    '''
    
  
    if roi.shape[0] <= template.shape[0] or roi.shape[1]<= template.shape[1]:
               
           #template is bigger then roi. Can not perform matchTemplate

            empty_list=[]
            empty_array = np.array(empty_list)
            loc = (empty_array,empty_array )
    else:
            


            method = cv.TM_CCORR_NORMED
            res = cv.matchTemplate(roi,template,method) # try tofind the pulse using a template match
            # Getting the max
            # x, y = np.unravel_index(np.argmax(res), res.shape)
            # print("INFO: max correlation is {} in x = {} and y = {}.".format(np.max(res),x,y))

            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
 
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                 top_left = min_loc
                 x = top_left[1]
                 y = top_left[0]
                 similarity_value = min_val
                 print("INFO: min similarity value is {} in x = {} and y = {}.".format(min_val,x,y))
            else:
                 top_left = max_loc
                 x = top_left[1]
                 y = top_left[0]
                 similarity_value = max_val
                 print("INFO: max similarity value is {} in x = {} and y = {}.".format(max_val,x,y))
            #bottom_right = (top_left[0] + w, top_left[1] + h)

            x = top_left[1]
            y = top_left[0]
            
            template_width, template_height = template.shape
            if verbose > 1:
                plt.imshow(roi)
                rect = plt.Rectangle((y, x), template_height, template_width, color='red', 
                     fc='none')
                plt.gca().add_patch(rect)
                plt.title('Grayscale Image with Bounding Box around the pulse')
                plt.show()
            
            loc = np.where( res >= threshold)
           
    if len(loc[0])>0:
        detected = True # pulse was detected 

        ppts=np.array(list(map(list, zip(*loc[::-1])))) #obtain um array from the list of tuples
        #print(ppts)
        ppts_max= ppts[:,0].max()
        ppts_min= ppts[:,0].min()
        ppts_median = np.median(ppts[:,0])
        #print(ppts_max, ppts_median, ppts_min)
        
        # extracted_pulse = roi[:,ppts_max:ppts_max+w]
        # baseline_pulse = np.argmax(np.std(extracted_pulse, axis =1))
        # _,_,xpulse,ypulse= get_values_from_img(extracted_pulse)
        # wpulse,hpulse = measure_extract_pulse(xpulse,ypulse)
        
        extracted_pulse = roi[x:x+template_width,y:y+template_height]
        # plt.imshow(extracted_pulse)
        # plt.show()
        _,_,xpulse,ypulse= get_values_from_img(extracted_pulse)
        wpulse,hpulse = measure_extract_pulse(xpulse,ypulse)              
              
    else:
              # There was a pulse to be detected but the detection failed
              # No pulse detected or the roi has no pulse
              detected = False
              #curve_scales.append((np.nan,np.nan))
              wpulse = np.nan
              hpulse = np.nan
              
    return detected,location, similarity_value, x,y, wpulse, hpulse, 

def print_segment_list(segment_list):
     for seg in segment_list:
          print("line number: {} - name: {} - segment length:{}".format (seg['line'],seg['name'] ,seg['lseg']))
          fig = plt.figure()
          #ax = fig.gca()
          #ax.set_xticks(np.arange(0, 1.1*np.max(seg['xseg']), 0.1))
          #ax.set_yticks(np.arange(0, 1.5 *np.max(seg['yseg']), 0.1))
          plt.title(seg['name'])
          plt.plot(seg['xseg'],seg['yseg'])
          plt.grid()
          plt.show()
          
def print_line_dict(line):
     for key, value in line.items():
      if key=='curves':
           print_segment_list(value)
      else:   print(f"{key}: {value}")

def interpolate_segment(x,y,num):
     x_interp = np.linspace(0.0,1.0,len(x))
     f = interpolate.CubicSpline(x_interp, y)
     x_new = np.linspace(0.0, 1.0, int(num))
     y_new = f(x_new)
     return x_new,y_new

def segment_to_df(line_list, pulse_per_sec, pulse_per_mv,num_pts):
     '''
        INPUT:
                line_list
                pulse_per_sec
                pulse_per_mv
                num_pts: number of points after the interpolation
     '''
     
     df = pd.DataFrame()
     for line in line_list:
          for seg in (line['curves']):
               xsec,ymv= convert_to_secmv(seg['xseg'],seg['yseg'],line['wpulse'],
                                           line['hpulse'],seg['wseg'], seg['baseline'], pulse_per_sec, pulse_per_mv)
               x_new,y_new =  interpolate_segment(xsec,ymv,num_pts)
               df[seg['name']]=y_new

     return df 

     

def remove_text(image, confidence_threshold):
     image_copy = image.copy()  
     results = pytesseract.image_to_data(image_copy, config='--psm 11',output_type='dict')

     for i in range(len(results["text"])):
         # extract the bounding box coordinates of the text region from
         # the current result
       x = results["left"][i]
       y = results["top"][i]
       w = results["width"][i]
       h = results["height"][i]
       # Extract the confidence of the text
       conf = int(results["conf"][i])
    
       if conf > 100*confidence_threshold: # adjust to your liking
        # Cover the text with a black rectangle
        print("INFO: word detect in the image")
        cv.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 0), -1)  
     return image_copy   
          
def process_line(line_number, labeled_line,offset,line_leads,config_dict, verbose = 0) :
    ''' 
    
    '''
    # TODO: Clean this dictonary
    line_dict ={}
    line_dict['wpulse']=config_dict['wpulse']
    line_dict['hpulse']= config_dict['hpulse']
    line_dict['curves']=[]
    line_dict['offset_line'] = offset

    # if verbose > 1: 
    #      display_segments("Labeled Line", labeled_line)

    u,c = np.unique(labeled_line, return_counts=True) 
    segment_labels = np.argsort(-c[1:])+1  # sort label by segment size in decresent order
    segment_length = -np.sort(-c[1:])
    max_label = np.max(u)

    app_seg_size = labeled_line.shape[1]//config_dict['layout'][1] 
    if verbose > 2:
            print("INFO: unique label {}.".format(u))
            print("INFO: count {}.".format(c))
            print("INFO: segment labels {}.".format(segment_labels))
            print("INFO: segment lenghth {}.".format(segment_length))
            
    larger_segments = segment_labels[:config_dict['layout'][1]+1]
   
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
            print("INFO: label = {},length = {} and ratio = {} .".format(label,roi_copy.shape[1], ratio ))
            plt.imshow(roi_copy)
            plt.show()

        ratio = round((length/app_seg_size),0)
        segment_ratios.append(ratio)

        # Based on the ratio, classifiy the segments
        #if ratio > 1 then split the segments 

        if ratio == 4.0:
            print("INFO: Four Segments {}.".format(ratio))
            print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

            if line_number+1 == config_dict['rhythm']: # line number : 0,1,...
                # if this line a rhythm line the the type is rhythm and just copy to  curves
                print("INFO: Line number {} and layout lines {} .".format(line_number,config_dict['layout'][0]))
                print("INFO: One Segment {}.".format(ratio))
                print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

                slx_seg1_start = sl[0][0].start
                slx_seg1_stop =  sl[0][0].stop
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

                #Take the slice from the labeled line
                seg = labeled_line[slice(*(segment_dict ['start_x'],segment_dict ['stop_x'], None)),
                               slice(*(segment_dict ['start_y'],segment_dict ['stop_y'], None))]
            
                seg = np.where(seg==label,255,0)
                #seg = seg.astype("uint8")
                if verbose > 1 :
                     title = "line: "+str(line_number) + "segment: " + str(label)
                     plt.imshow(seg)
                     plt.title(title)
                     plt.show() 

                ws, ls,xs,ys= get_values_from_img(seg)
                segment_dict ['wseg']=ws 
                segment_dict ['lseg']=ls 
                segment_dict ['xseg']=xs 
                segment_dict ['yseg']=ys

                baseline = np.argmax(np.std(seg, axis =1))
                segment_dict ['baseline']= baseline


                print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict ['lseg']))

            else:
                print("INFO: Four Segments")

        elif ratio == 3.0:
            print("INFO: Three Segments {}.".format(ratio))
            print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

            # Separate the segments 

             # First segment
            slx_seg1_start = sl[0][0].start
            slx_seg1_stop =  sl[0][0].stop
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

            #Take the slice from the labeled line
            seg = labeled_line[slice(*(segment_dict ['start_x'],segment_dict ['stop_x'], None)),
                               slice(*(segment_dict ['start_y'],segment_dict ['stop_y'], None))]
            
            seg = np.where(seg==label,255,0)
            #seg = seg.astype("uint8")

            if verbose > 1 :
                title = "line: "+str(line_number) + "segment: " + str(label)
                plt.imshow(seg)
                plt.title(title)
                plt.show() 
           

            ws, ls,xs,ys= get_values_from_img(seg)
            segment_dict ['wseg']=ws 
            segment_dict ['lseg']=ls 
            segment_dict ['xseg']=xs 
            segment_dict ['yseg']=ys

           

            baseline = np.argmax(np.std(seg, axis =1))
            segment_dict ['baseline']= baseline

            # xsec,ymv= convert_to_secmv(xs,ys,line_dict['wpulse'], line_dict['hpulse'],
            #                            ws, baseline, config_dict['pulse_per_sec'],
            #                            config_dict['pulse_per_mv'])
            # segment_dict ['xseg']=xsec 
            # segment_dict ['yseg']=ymv

            line_dict['curves'].append(segment_dict)

            print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict ['lseg']))

            # Second Segment
 

            slx_seg2_start = sl[0][0].start
            slx_seg2_stop =  sl[0][0].stop 
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

            #Take the slice from the labeled line
            seg = labeled_line[slice(*(segment_dict ['start_x'],segment_dict ['stop_x'], None)),
                               slice(*(segment_dict ['start_y'],segment_dict ['stop_y'], None))]
            
            seg = np.where(seg==label,255,0)
            #seg = seg.astype("uint8") 

            if verbose > 1 :
                title = "line: "+str(line_number) + "segment: " + str(label)
                plt.imshow(seg)
                plt.title(title)
                plt.show() 

            ws, ls,xs,ys= get_values_from_img(seg)
            segment_dict ['wseg']=ws 
            segment_dict ['lseg']=ls 
            segment_dict ['xseg']=xs 
            segment_dict ['yseg']=ys

            baseline = np.argmax(np.std(seg, axis =1))
            segment_dict ['baseline']= baseline

            # xsec,ymv= convert_to_secmv(xs,ys,line_dict['wpulse'], line_dict['hpulse'],
            #                            ws, baseline, config_dict['pulse_per_sec'],
            #                            config_dict['pulse_per_mv'])
            # segment_dict ['xseg']=xsec 
            # segment_dict ['yseg']=ymv

            line_dict['curves'].append(segment_dict)

            print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict ['lseg']))

            max_label = max_label + 1

            # Third Segment
            slx_seg3_start = sl[0][0].start
            slx_seg3_stop =  sl[0][0].stop 
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

            #Take the slice from the labeled line
            seg = labeled_line[slice(*(segment_dict ['start_x'],segment_dict ['stop_x'], None)),
                               slice(*(segment_dict ['start_y'],segment_dict ['stop_y'], None))]
            
            seg = np.where(seg==label,255,0)
            #seg = seg.astype("uint8")

            if verbose > 1 :
                title = "line: "+str(line_number) + "segment: " + str(label)
                plt.imshow(seg)
                plt.title(title)
                plt.show() 
            
            # get the x,y values from the image
            ws, ls,xs,ys= get_values_from_img(seg)
            segment_dict ['wseg']=ws 
            segment_dict ['lseg']=ls 
            segment_dict ['xseg']=xs 
            segment_dict ['yseg']=ys


            baseline = np.argmax(np.std(seg, axis =1))
            segment_dict ['baseline']= baseline

            # xsec,ymv= convert_to_secmv(xs,ys,line_dict['wpulse'], line_dict['hpulse'],
            #                            ws, baseline, config_dict['pulse_per_sec'],
            #                            config_dict['pulse_per_mv'])
            # segment_dict ['xseg']=xsec 
            # segment_dict ['yseg']=ymv

            line_dict['curves'].append(segment_dict)

            print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict ['lseg']))


        elif ratio == 2.0:

            print("INFO: two  segments {} Slice X = {} and Slice Y =  {}" .format(ratio, sl[0][0], sl[0][1]) )

            # Separate the segments 

            # First segment
            slx_seg1_start = sl[0][0].start
            slx_seg1_stop =  sl[0][0].stop
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

            #Take the slice from the labeled line
            seg = labeled_line[slice(*(segment_dict ['start_x'],segment_dict ['stop_x'], None)),
                               slice(*(segment_dict ['start_y'],segment_dict ['stop_y'], None))]
            
            seg = np.where(seg==label,255,0)
            #seg = seg.astype("uint8")

            if verbose > 1 :
                title = "line: "+str(line_number) + "segment: " + str(label)
                plt.imshow(seg)
                plt.title(title)
                plt.show() 

            ws, ls,xs,ys= get_values_from_img(seg)
            segment_dict ['wseg']=ws 
            segment_dict ['lseg']=ls 
            segment_dict ['xseg']=xs 
            segment_dict ['yseg']=ys

            baseline = np.argmax(np.std(seg, axis =1))
            segment_dict ['baseline']= baseline

            # xsec,ymv= convert_to_secmv(xs,ys,line_dict['wpulse'], line_dict['hpulse'],
            #                            ws, baseline, config_dict['pulse_per_sec'],
            #                            config_dict['pulse_per_mv'])
            # segment_dict ['xseg']=xsec 
            # segment_dict ['yseg']=ymv

            line_dict['curves'].append(segment_dict)

            print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict ['lseg']))

            # Second segments 

            slx_seg2_start = sl[0][0].start
            slx_seg2_stop =  sl[0][0].stop 
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

            #Take the slice from the labeled line
            seg = labeled_line[slice(*(segment_dict ['start_x'],segment_dict ['stop_x'], None)),
                               slice(*(segment_dict ['start_y'],segment_dict ['stop_y'], None))]
            
            seg = np.where(seg==label,255,0)
            # seg = seg.astype("uint8")

            if verbose > 1 :
                title = "line: "+ str(line_number) + " segment : " + str(label)
                plt.imshow(seg)
                plt.title(title)
                plt.show() 
    

            ws, ls,xs,ys= get_values_from_img(seg)
            segment_dict ['wseg']=ws 
            segment_dict ['lseg']=ls 
            segment_dict ['xseg']=xs 
            segment_dict ['yseg']=ys


            baseline = np.argmax(np.std(seg, axis =1))
            segment_dict ['baseline']= baseline

            # xsec,ymv= convert_to_secmv(xs,ys,line_dict['wpulse'], line_dict['hpulse'],
            #                            ws, baseline, config_dict['pulse_per_sec'],
            #                            config_dict['pulse_per_mv'])
            # segment_dict ['xseg']=xsec 
            # segment_dict ['yseg']=ymv

            line_dict['curves'].append(segment_dict)

            print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict ['lseg']))

        elif ratio == 1.0:
            print("INFO: One Segment {}.".format(ratio))
            print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

            slx_seg1_start = sl[0][0].start
            slx_seg1_stop =   sl[0][0].stop
            sly_seg1_start =  sl[0][1].start
            sly_seg1_stop =   sl[0][1].stop

           
            # append segment to the segment list

            #segment_dict = fill_slice_info(line_number,label, slx_seg1_start, slx_seg1_stop,sly_seg1_start, sly_seg1_stop)
            segment_dict = {}
            segment_dict['line'] = line_number
            segment_dict['label'] =label
            segment_dict ['start_x'] = slx_seg1_start
            segment_dict ['stop_x'] = slx_seg1_stop
            segment_dict ['start_y'] = sly_seg1_start
            segment_dict ['stop_y'] = sly_seg1_stop

            #Take the slice from the labeled line
            seg = labeled_line[slice(*(segment_dict ['start_x'],segment_dict ['stop_x'], None)),
                               slice(*(segment_dict ['start_y'],segment_dict ['stop_y'], None))]
            
            seg = np.where(seg==label,255,0)
            #seg = seg.astype("uint8")
            # seg= remove_text(seg,0.6)
            # title = "line: "+str(line_number) + "segment: " + str(label)
            # plt.imshow(seg)
            # plt.title(title)
            
            if verbose > 1 :
                title = "line: "+str(line_number) + "segment: " + str(label)
                plt.imshow(seg)
                plt.title(title)
               
                plt.show() 
            
            ws, ls,xs,ys= get_values_from_img(seg)
            segment_dict ['wseg']=ws 
            segment_dict ['lseg']=ls 
            segment_dict ['xseg']=xs 
            segment_dict ['yseg']=ys
            
            # seen = []
            # result=[]

          
            
            # for i, t in enumerate(zip(xs,ys)):
             
            #     if t[0] not in seen:
            #         seen.append(t[0])
            #         result.append(t)
            #     else:
            #         print("DEBUG: repetead x = {}  t ={}".format(t[0],t))

            baseline = np.argmax(np.std(seg, axis =1))
            segment_dict ['baseline']= baseline


            # xsec,ymv= convert_to_secmv(xs,ys,line_dict['wpulse'], line_dict['hpulse'],
            #                            ws, baseline, config_dict['pulse_per_sec'],
            #                            config_dict['pulse_per_mv'])
            # segment_dict ['xseg']=xsec 
            # segment_dict ['yseg']=ymv

            print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict ['lseg']))
            
            # fig = plt.figure()
            # ax = fig.gca()
            # ax.set_xticks(np.arange(0, 1.1*np.max(xsec), 0.1))
            # ax.set_yticks(np.arange(0, 1.5 *np.max(ymv), 0.1))
            # plt.plot(xsec,ymv)
            # plt.grid()
            # plt.show()

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

                     if is_nan(config_dict['wpulse']):
                          print("INFO:No check with template is possible")
                          print("INFO: pulse width: {} and pulse height {}".format(wpulse, hpulse))
                     else:
                          if wpulse-10 <= config_dict['wpulse'] and wpulse+10 >=config_dict['wpulse']:
                               print("INFO: pulse width checked {} and {}".format(wpulse, config_dict['wpulse']))

                          if hpulse-10 <= config_dict['hpulse'] and hpulse+10 >=config_dict['hpulse']:
                               print("INFO: pulse height checked {} and {}".format(hpulse, config_dict['hpulse']))
                    
                
                     line_dict ['wpulse'] = wpulse
                     line_dict ['hpulse'] = hpulse
                else:
                     #if it is no suppose to have a pulse, then it is garbage
                     break
               

        if ratio < smallest_ratio:
            smallest_ratio = ratio
    
    line_dict['curves'] = sorted(line_dict['curves'], key=lambda d: d['start_y'])

    
    for i, d in enumerate(line_dict['curves']):
        d['name'] = line_leads[i]  #add  the name of the segments

    
    
    return line_dict

   


                 
