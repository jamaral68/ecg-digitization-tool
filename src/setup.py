import cv2
import sys

def setup_ecg(layout, pulse, rhythm, image_name):
    """
    Complete configuration for ECG processing
    """
    
    # the names dependending on the layout
    if layout[1]== 4 and layout[0]==3:
        lt_leads = [ ['I', 'aVR','V1','V4'],
                        ['II','aVL','V2','V5'],
                        ['III','aVF', 'V3','V6'],
                        ['II']

                        ]
    elif layout[1]==2:
        raise NotImplementedError ('Not implemented' )
    elif layout[1]==1:

       # raise NotImplementedError ('Not implemented' )
       lt_leads = [ ['I'],
                    ['II'],
                    ['III'],
                    ['aVR'],
                    ['aVL'],
                    ['aVF'],
                    ['V1'],
                    ['V2'],
                    ['V3'],
                    ['V4'],
                    ['V5'],
                    ['V6']]

    else:
        raise ValueError('columns must be 4, 2 or 1')
    
    # Define pulse detection
    if pulse == 0 :
        print("INFO: No pulse to be detected")
    elif pulse == -1:
        print("INFO: pulse to be detected in all lines")
    elif  isinstance(pulse, list): 
        print("INFO: pulse on lines: {}.".format(pulse)) 
        for p in pulse:
            lt_leads[p].append('Pulse')
    elif isinstance(pulse, int): 
        print("INFO: pulse on line: {}.".format(pulse)) 
        lt_leads[pulse].append('Pulse')

    else:
        raise ValueError('pulse should  be 0, an int or a list')
    
    # Define rhythm
    if rhythm == 0:
        print("INFO: No rhythm lead") 
    else:
        print("INFO: rhythm on line : {}.".format(rhythm)) 

    #load the image 
    image = cv2.imread(image_name)

    # sanity check
    if image is None:
        print('Cannot open image: ' + image_name)
        sys.exit(0)
    
    return lt_leads, image