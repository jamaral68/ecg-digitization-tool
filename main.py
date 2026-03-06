from edt import ecg_to_csv
from edt_utils import plot_ecg
from matplotlib import pyplot as plt

# Main program 
filename = 'ecg_test2'
image_name = filename + '.png'
template_name = 'pul.png'

csv_name = filename + '.csv'
layout = (3,4)
pulse = [0,1,2]  
rhythm = 4 # which line has the rhythum
verbose = 6
mmpsec = 25 # 25 mm/seg
mmpmv = 10 # 10 mm/mV
pulse_width_mm = 5 # pulse width in mm
pulse_height_mm =10  # pulse height in mm
pulse_per_sec = pulse_width_mm/mmpsec
pulse_per_mv= pulse_height_mm/mmpmv
sample_frequency = 500
time_lead = 2.5 # duratiom of the segment in seconds
num_sampling_points = time_lead/(1/sample_frequency)
location = 'right'
strategy = 'none'  # It can be filter or color
lower=(0,0,0) # black color
upper=(179,255,220) # dark gray
thres_value = 127
kSize2d = 3 
kSize1d = 3
perc_space_leads =0.2
dilation = 10
perc_max_dist = 0.7 

config_dict ={}
config_dict['pulse'] = pulse # which lines have pulse
config_dict['rhythm'] = rhythm # which row has the rhythm signal
config_dict['verbose'] = verbose 
config_dict['mmpsec']= mmpsec
config_dict['mmpmv']=mmpmv
config_dict['pulse_width_mm'] = pulse_width_mm
config_dict['pulse_height_mm'] = pulse_height_mm
config_dict['pulse_per_sec'] = pulse_per_sec
config_dict['pulse_per_mv'] = pulse_per_mv
config_dict['sample_frequency'] = sample_frequency
config_dict['time_lead'] = time_lead
config_dict['location']= location
config_dict['layout']=layout   # tuple with the layout    
config_dict['pulse_width_mm']  = pulse_width_mm
config_dict['pulse_height_mm'] = pulse_height_mm
config_dict['pulse_per_mv']= pulse_per_mv
config_dict['pulse_per_sec']= pulse_per_sec
config_dict['num_sampling_points']= num_sampling_points
config_dict['strategy'] = strategy
config_dict['lower']= lower
config_dict['upper']= upper
config_dict['thres_value'] = thres_value
config_dict['kSized2d'] = kSize2d
config_dict['kSized1d'] = kSize1d
config_dict['perc_space_leads'] = perc_space_leads
config_dict['dilation'] = dilation
config_dict['perc_max_dist'] = perc_max_dist 

df=ecg_to_csv(image_name ,template_name, csv_name, config_dict )

# Plot in the lay out
plot_ecg(df,df.columns,csv_name, n_rows = layout[0], n_columns = layout[1], fs = 500, figure_size = (20, 12))
plt.show()

print("THE END")