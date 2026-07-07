import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
from enum import Enum

# Initialize indexing scheme for the LED model
vert_multiples  = [    3,2,    7,6,    11,10,      15,14,      19,18,      23,22,      27,26,      31,30,      35,34,      39,38,      43,42,      47,46,      51,50,      55,54,      59,58,     63,62]
horiz_multiples = [0,1,    4,5,    8,9,      12,13,      16,17,      20,21,      24,25,      28,29,      32,33,      36,37,      40,41,      44,45,      48,49,      52,53,      56,57,      60,61     ]
vert_multiples.reverse()
horiz_multiples.reverse()
vert_LED_inds = []
horiz_LED_inds = []
top_vert_inds_row = np.multiply(vert_multiples, 16)
for i in range(16):
    vert_LED_inds.append((top_vert_inds_row - i + 15).tolist())
top_horiz_inds_row = np.multiply(horiz_multiples, 16)
for i in range(16):
    horiz_LED_inds.append((top_horiz_inds_row + i).tolist())
vert_LED_inds = np.array(vert_LED_inds).T
horiz_LED_inds = np.array(horiz_LED_inds).T

# Create planar indexing chunks for horizontally sequential lighting (beam events)
beam_index_list = []
for base in range(960, -1, -64):
    chunks = [
        list(range(base+15, base-1, -1)), # chunk 0
        list(range(base+31, base+15, -1)), # chunk 1
        list(range(base+47, base+31, -1)), # chunk 2
        list(range(base+63, base+47, -1)) # chunk 3
    ]
    beam_index_list.extend(chunks[2])
    beam_index_list.extend(chunks[1])
    beam_index_list.extend(chunks[3])
    beam_index_list.extend(chunks[0])

# Create planar indexing chunks for vertically sequential lighting (cosmic events)
cosmic_index_list = []
for base in range(960, -1, -64):
    chunk_0 = list(range(base+47, base+31, -1))
    chunk_1 = list(range(base+63, base+47, -1))
    # The following two lines were formerly reversed in position. This is now believed to be correct, but this comment is left here as a REMINDER IF NOT WORKING
    cosmic_index_list.extend(chunk_0)
    cosmic_index_list.extend(chunk_1)


# Set the static color to be used if full color events are disabled
static_color = 0x006600

# Initialize sizes of various types of arrays used throughout
ARRAY_SHAPE = (160, 80) # Simulation is processed into arrays of this shape (same proportions as LED model)
LED_SHAPE = (32, 16) # Physical LED model shape

# Region of pixels grouped together to compress, turns out to be 5x5 for the given shapes above
BLOCK_Y = ARRAY_SHAPE[0] // LED_SHAPE[0]
BLOCK_X = ARRAY_SHAPE[1] // LED_SHAPE[1]

# Set the normalization for the color scale 
# WARNING: vmax is manually set to be the max energy acoss all events in the display set
vmax = 946.5 # max energy displayed on color scale
vmin = 10 # small nonzero value to avoid errors resulting from zero
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

# Create enum for classifying event types
class EventType(Enum):
    NUMU_QE = 0
    NUMU_RES = 1
    NUMU_DIS = 2
    NUMU_OTHER = 3
    NUE_QE = 4
    NUE_RES = 5
    NUE_DIS = 6
    NUE_OTHER = 7
    NC = 8
    COSMIC = 9



class Event:
    '''
    Event class constructor:
        Event(horiz_array, vert_array, event_type)
            horiz_array : 2D array of horiz event information
            vert_array : 2D array of vert event information
            event_type : string, name of type of interaction which produced the event
    Event class attributes:
        .horiz : 2D array of horiz event information, shape and contents modified by various methods
        .vert : 2D array of vert event information, shape and contents modified by various methods
        .event_type : string, name of type of interaction which produced the event
        WARNING: this attribute is not initialized by the constructor; it comes into being after executing the method .led_output()
        .led_array : 1D array of hex colors, facilitates handoff of event information to physical model
    Event class methods:
        .crop_event(cutoff=True, threshold=0.01)
            Description:
                reduces size of self.horiz and self.vert to contain only a box around pixels with energy above specified cutoff
            Arguments:
                cutoff : boolean, tells whether events below threshold will be removed
                threshold : float, 0 <= threshold < 1, fraction of max energy below cutoff will be enacted
        .stretch_event(enable_random=False, min_random=0.5, max_random=1)
            Description:
                resizes self.horiz and self.vert to stretched size proportional to scale of LED model or to random factors smaller subject to constraints of arguments
            Arguments:
                enable_random : boolean, tells whether randomized stretch factor is enabled
                min_random : float, 0 < min_random <= 1, min possible randomized stretch factor if enabled
                max_random : float, min_random <= max_random <= 1, max possible randomized stretch factor if enabled
        .pad_event(x_offset_horiz=0, x_offset_vert=0, y_offset=0)
            Description:
                resizes self.horiz and self.vert to 2D arrays of size proportional to scale of LED model with pixels surrounding initially populated region set to 0, subject positional offsets in arguments
            Arguments:
                x_offset_horiz : integer, number of rows of array to offset active region of self.horiz[1] before filling with zeros
                x_offset_vert : integer, number of rows of array to offset active region of self.vert[1] before fillingt with zeros
                y_offset : integer, number of rows of array to offset active region of self.horiz[0] and self.vert[0] before filling with zeros
        .compresss_to_led_grid(mode="sum")
            Description:
                resizes self.horiz and self.vert to size exactly matching the number of LEDs in the horiz and vert planes of the physical LED model
            Arguments:
                mode : string, identifies method of compression, must take values 'sum' or 'max' or 'mean'
        .values_to_hex(cmap_name="rainbow", threshold=0.01, brightness=96.0/255.0)
            Description:
                converts all entries of self.vert and self.horiz to hex colors based on the color map
            Arguments:
                cmap_name : string, name of the matplotlib cmap option chosen for LEDs to display
                threshold : float, 0 <= threshold <= 1, fraction of max energy beneath which LEDs will not be lit
                brightness : float, 0 <= threshold <= 1, fraction of max brightness desired for LEDs to emit
        .led_output()
            Description:
                creates the self.led_array attribute, a 1D array consisting of the contents of self.horiz and self.vert in an order matching the indexing of the physical LED model
            Arguments:
                None
        .display(strip, mode="progressive", color_enabled=True)
            Description:
                lights up the physical LED model according to the mode specified by the argument
            Arguments:
                strip : apa102 light strip to be lit according to the event
                mode : string, identifies method of turning on LEDs, must take values 'progressive' or 'full' or 'planar' or 'fade'
                color_enabled : boolean, tells whether full color outputs will be used as opposed to mono-color outputs
    '''

    # "Event" class constructor; takes 2D horiz array, 2D vert array, string event type
    def __init__(self, horiz_array, vert_array, event_type):
        self.horiz = horiz_array
        self.vert = vert_array
        self.event_type = event_type

    # Resize event to box surrounding active pixels
    def crop_event(self, cutoff=True, threshold=0.01):
        # Excise energy values below threshold from cropping consideration
        # If no cutoff is desired, excise nothing
        if cutoff == False:
            threshold = 0
        # Raise error if cutoff threshold is not valid
        elif cutoff == True: 
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("'threshold' must be float between 0 and 1")
        # Raise error if choice to cutoff is unclear
        else:
            raise ValueError("'cutoff' must be boolean True or False")
        horiz_max_value = max(max(row) for row in self.horiz)
        vert_max_value = max(max(row) for row in self.vert)
        horiz_threshold = threshold*horiz_max_value
        vert_threshold = threshold*vert_max_value
        mask = (self.horiz > horiz_threshold) | (self.vert > vert_threshold)
        y_coords, x_coords = np.where(mask)
        # Crop event object to smaller size, rectangle surrounding active pixels
        self.horiz = self.horiz[y_coords.min():y_coords.max()+1, x_coords.min():x_coords.max()+1]
        self.vert = self.vert[y_coords.min():y_coords.max()+1, x_coords.min():x_coords.max()+1]

    # Resize event to size proportional to the physical LED model
    def stretch_event(self, enable_random=False, min_random=0.5, max_random=1):
        # Set random scale factors for each dimension within specified range if desired
        if enable_random == True:
            if not (0.0 < min_random <= 1.0):
                raise ValueError("'min_random' must be float between 0 and 1")
            if not (min_random <= max_random <= 1.0):
                raise ValueError("'max_random' must be float greater than min_random between 0 and 1")
            scale_y = np.random.uniform(min_random, max_random)
            scale_x_horiz = np.random.uniform(min_random, max_random)
            scale_x_vert = np.random.uniform(min_random, max_random)
        # Full stretch if no scale factor provided
        elif enable_random == False:
            scale_y = 1
            scale_x_horiz = 1
            scale_x_vert = 1
        # Raise error if provided stretch factor is not valid
        else:
            raise ValueError("'enable_random' must be boolean True or False")
        # Stretch event arrays to full array size (proportional to model size) in accordance with scale factors, or to random factor smaller
        horiz_zoomed = zoom(self.horiz, (ARRAY_SHAPE[0]*scale_y/self.horiz.shape[0], ARRAY_SHAPE[1]*scale_x_horiz/self.horiz.shape[1]), order=1)
        vert_zoomed = zoom(self.vert, (ARRAY_SHAPE[0]*scale_y/self.vert.shape[0], ARRAY_SHAPE[1]*scale_x_vert/self.vert.shape[1]), order=1)
        # Impose energy conservation and update event to the stretched shape
        self.horiz = horiz_zoomed*((self.horiz).sum()/horiz_zoomed.sum())
        self.vert = vert_zoomed*((self.vert).sum()/vert_zoomed.sum())

    def pad_event(self, x_offset_horiz=0, x_offset_vert=0, y_offset=0):
        # Verify integer offsets
        for offset in (x_offset_horiz, x_offset_vert, y_offset):
            if not isinstance(offset, (int, np.integer)):
                raise ValueError("'x_offset_horiz', 'x_offset_vert', and 'y_offset' must all be integers")
        # Define helper function for overlap test
        def overlaps(a0, a1, b0, b1):
            return (a0 < b1) and (a1 > b0)
        # Extract event dimensions
        h, w = self.horiz.shape
        # Detector-volume overlap test
        inside_x_horiz = overlaps(x_offset_horiz, x_offset_horiz + w, 0, ARRAY_SHAPE[1])
        inside_x_vert = overlaps(x_offset_vert, x_offset_vert + w, 0, ARRAY_SHAPE[1])
        inside_y = overlaps(y_offset, y_offset + h, 0, ARRAY_SHAPE[0])
        # If the event volume does not overlap the detector in ALL THREE dimensions, suppress both views.
        if not (inside_x_horiz and inside_x_vert and inside_y):
            self.horiz = np.zeros(ARRAY_SHAPE)
            self.vert = np.zeros(ARRAY_SHAPE)
            return
        # Define helper function for clipped placement
        def clip_into_detector(src, x_offset, y_offset):
            # Extract event dimensions
            H, W = ARRAY_SHAPE
            h, w = src.shape
            # Initialize output array
            output = np.zeros((H, W))
            # Destination coordinates within detector
            output_x0 = max(0, x_offset)
            output_y0 = max(0, y_offset)
            output_x1 = min(W, x_offset + w)
            output_y1 = min(H, y_offset + h)
            # No overlap for this projection
            if output_x1 <= output_x0 or output_y1 <= output_y0:
                return output
            # Corresponding source coordinates
            src_x0 = max(0, -x_offset)
            src_y0 = max(0, -y_offset)
            src_x1 = src_x0 + (output_x1 - output_x0)
            src_y1 = src_y0 + (output_y1 - output_y0)
            output[output_y0:output_y1, output_x0:output_x1] = src[src_y0:src_y1, src_x0:src_x1]
            return output
        # Place both projections with clipping
        self.horiz = clip_into_detector(self.horiz, x_offset_horiz, y_offset)
        self.vert = clip_into_detector(self.vert, x_offset_vert, y_offset)

    # Compress an event with the same proportions as the model to the size of the model
    # WARNING: Behavior for events not scaled to the same proportions as the model is untested
    def compress_to_led_grid(self, mode="sum"):
        # Reshape event to size of the LED model
        reshaped_horiz = (self.horiz).reshape(LED_SHAPE[0], BLOCK_Y, LED_SHAPE[1], BLOCK_X)
        reshaped_vert = (self.vert).reshape(LED_SHAPE[0], BLOCK_Y, LED_SHAPE[1], BLOCK_X)
        # Allows choice to select sum, max, or mean of each block based on mode
        if mode == "sum":
            self.horiz = reshaped_horiz.sum(axis=(1,3))
            self.vert = reshaped_vert.sum(axis=(1,3))
        elif mode == "max":
            self.horiz = reshaped_horiz.max(axis=(1,3))
            self.vert = reshaped_vert.max(axis=(1,3))
        elif mode == "mean":
            self.horiz = reshaped_horiz.mean(axis=(1,3))
            self.vert = reshaped_vert.mean(axis=(1,3))
        else:
            raise ValueError("mode must be 'sum' or 'max' or 'mean'")

    # Convert event information from energy to color hex codes
    def values_to_hex(self, cmap_name="jet", threshold=0.01, brightness=96.0/255.0):
        # Raise error if provided threshold is invalid
        if threshold < 0 or threshold > 1:
            raise ValueError("'threshold' must be float between 0 and 1")
        # Raise error if provided brightness is invalid
        if brightness < 0 or threshold > 1:
            raise ValueError("'brightness' must be float between 0 and 1")
        # Convert horiz array from energies to colors
        cmap = plt.get_cmap(cmap_name)
        for i in range(len(self.horiz)):
            for j in range(len(self.horiz[0])):
                if self.horiz[i][j] < threshold*vmax:
                    rgb = 0
                else:
                    r, g, b, _ = cmap(norm(self.horiz[i][j]))
                    r = int(r*255*brightness)
                    g = int(g*255*brightness)
                    b = int(b*255*brightness)
                    rgb = (r << 16) | (g << 8) | b
                self.horiz[i][j] = rgb
        # Convert vert array from energies to colors
        for i in range(len(self.vert)):
            for j in range(len(self.vert[0])):
                if self.vert[i][j] < threshold*vmax:
                    rgb = 0
                else: 
                    r, g, b, _ = cmap(norm(self.vert[i][j]))
                    r = int(r*255*brightness)
                    g = int(g*255*brightness)
                    b = int(b*255*brightness)
                    rgb = (r << 16) | (g << 8) | b
                self.vert[i][j] = rgb

    # Assigns the "led_array" attribute based on current values of "horiz" and "vert" attributes
    def led_output(self):
        # Initialize output list
        led_list = []
        rows, cols = (self.vert).shape
        # Horizontal plane
        for y in range(rows):
            for x in range(cols):
                led_index = horiz_LED_inds[y,x]
                value = self.horiz[y,x]
                led_list.append([led_index, value])
        # Vertical plane
        for y in range(rows):
            for x in range(cols):
                led_index = vert_LED_inds[y,x]
                value = self.vert[y,x]
                led_list.append([led_index, value])
        # Store output as array instead of list
        led_array = np.array(led_list)
        # Sort by LED index so it matches physical strip order
        led_array = led_array[led_array[:,0].argsort()]
        self.led_array = led_array

    # Outputs event to LED model according to provided mode
    def display(self, strip, mode="progressive", color_enabled=True, num_led=1024):
        # Display event in progressively filled layers
        if mode == "progressive":
            # Progressively fill horizontal layers for cosmics
            if self.event_type == EventType.COSMIC:
                # Assign all LED outputs
                for i in range(len(cosmic_index_list)):
                    if self.led_array[cosmic_index_list[i]][1] != 0:
                        color = int(self.led_array[cosmic_index_list[i]][1])
                        if color_enabled == True:
                            strip.set_pixel_rgb(cosmic_index_list[i], color)
                        else:
                            strip.set_pixel_rgb(cosmic_index_list[i], static_color)
                    # Display LED outputs in layers
                    if i % 32 == 0:
                        # Light the adjacent LEDs on the horizontal plane
                        if (cosmic_index_list[i]+1) % 16 == 0:
                            left_index = cosmic_index_list[i] - 47
                            right_index = cosmic_index_list[i] - 63
                        else:
                            left_index = cosmic_index_list[i] + 17
                            right_index = cosmic_index_list[i] - 31
                        if left_index < 1024:
                            for j in range(0, 16):
                                color_left = int(self.led_array[left_index+j][1])
                                strip.set_pixel_rgb(left_index+j, color_left)
                        if right_index >= 0:
                            for j in range(0, 16):
                                color_right = int(self.led_array[right_index+j][1])
                                strip.set_pixel_rgb(right_index+j, color_right)
                        strip.show()
            # Progressively fill vertical layers for all events except cosmics
            else:
                # Assign all LED outputs 
                for i in range(len(self.led_array)):
                    if self.led_array[beam_index_list[i]][1] != 0:
                        color = int(self.led_array[beam_index_list[i]][1])
                        if color_enabled == True:
                            strip.set_pixel_rgb(beam_index_list[i], color)
                        else:
                            strip.set_pixel_rgb(beam_index_list[i], static_color)
                    # Display LED outputs in layers
                    if i % 16 == 0:
                        strip.show()
                strip.show()
        # Display entire event all at once
        elif mode == "full":
            for i in range(len(self.led_array)):
                if self.led_array[beam_index_list[i]][1] != 0:
                    color = int(self.led_array[beam_index_list[i]][1])
                    if color_enabled == True:
                        strip.set_pixel_rgb(beam_index_list[i], color)
                    else:
                        strip.set_pixel_rgb(beam_index_list[i], static_color)
            strip.show()
        # Display event in planes passing through the model
        elif mode == "planar":
            # Vertically sweep through planes for cosmics
            if self.event_type == EventType.COSMIC:
                # Assign all LED outputs
                for i in range(len(cosmic_index_list)):
                    if self.led_array[cosmic_index_list[i]][1] != 0:
                        color = int(self.led_array[cosmic_index_list[i]][1])
                        if color_enabled == True:
                            strip.set_pixel_rgb(cosmic_index_list[i], color)
                        else:
                            strip.set_pixel_rgb(cosmic_index_list[i], static_color)
                    # Display LED outputs in layers
                    if i % 32 == 0:
                        strip.show()
                        for j in range(i, i-16, -1):
                            strip.set_pixel_rgb(cosmic_index_list[j], 0x000000)
                        strip.show()
                strip.show()
            # Horizontally sweep through planes for all events except cosmics
            else:
                # Assign all LED outputs
                for i in range(len(self.led_array)):
                    if self.led_array[beam_index_list[i]][1] != 0:
                        color = int(self.led_array[beam_index_list[i]][1])
                        if color_enabled == True:
                            strip.set_pixel_rgb(beam_index_list[i], color)
                        else:
                            strip.set_pixel_rgb(beam_index_list[i], static_color)
                    # Display LED outputs in layers
                    if i % 16 == 0:
                        strip.show()
                        for j in range(i, i-16, -1):
                            strip.set_pixel_rgb(beam_index_list[j], 0x000000)
                        strip.show()
                strip.show()
        # Display event trajectories that progressively appear and fade away
        elif mode == "fade":
            # INCOMPLETE
            # Vertically sweep and fade for cosmics
            if self.event_type == EventType.COSMIC:
                # Initialize framebuffer
                current_colors = np.zeros(num_led)
                # Assign all initial LED outputs
                for i in range(len(cosmic_index_list)):
                    if self.led_array[cosmic_index_list[i]][1] != 0:
                        color = int(self.led_array[cosmic_index_list[i]][1])
                        if color_enabled == True:
                            current_colors[cosmic_index_list[i]] = color
                            strip.set_pixel_rgb(cosmic_index_list[i], color)
                        else:
                            current_colors[cosmic_index_list[i]] = static_color
                            strip.set_pixel_rgb(cosmic_index_list[i], static_color)
                    # Trun on lights at front column of trajectory
                    # Fade away lights from previous 6 columns of trajectory
                    if i % 32 == 0:
                        for j in range(i, i-32, -1):
                            for k in range(1,7):
                                update_index = j-32*k
                                if update_index >= len(cosmic_index_list):
                                    continue
                                rgb = int(current_colors[cosmic_index_list[update_index]])
                                r = (rgb >> 16) & 0xFF
                                g = (rgb >> 8) & 0xFF
                                b = rgb & 0xFF
                                fade = max(0, 1-0.15*k)
                                r = int(r*fade)
                                g = int(g*fade)
                                b = int(b*fade)
                                new_rgb = (r << 16) | (g << 8) | b
                                current_colors[cosmic_index_list[update_index]] = new_rgb
                                strip.set_pixel_rgb(cosmic_index_list[update_index], new_rgb)
                        strip.show()
                strip.show()
                # Turn off residual pixels after reaching end 
                for index in beam_index_list:
                    strip.set_pixel_rgb(index, 0x000000)
                strip.show()
            # Horizontally sweep and fade for all events except cosmics
            else:
                # Initialize framebuffer
                current_colors = np.zeros(num_led)
                # Assign all initial LED outputs
                for i in range(len(self.led_array)):
                    if self.led_array[beam_index_list[i]][1] != 0:
                        color = int(self.led_array[beam_index_list[i]][1])
                        if color_enabled == True:
                            current_colors[beam_index_list[i]] = color
                            strip.set_pixel_rgb(beam_index_list[i], color)
                        else:
                            current_colors[beam_index_list[i]] = static_color
                            strip.set_pixel_rgb(beam_index_list[i], static_color)
                    # Trun on lights at front column of trajectory
                    # Fade away lights from previous 6 columns of trajectory
                    if i % 16 == 0:
                        for j in range(i, i-16, -1):
                            for k in range(1,7):
                                update_index = j-16*k
                                if update_index >= len(beam_index_list):
                                    continue
                                rgb = int(current_colors[beam_index_list[update_index]])
                                r = (rgb >> 16) & 0xFF
                                g = (rgb >> 8) & 0xFF
                                b = rgb & 0xFF
                                fade = max(0, 1-0.15*k)
                                r = int(r*fade)
                                g = int(g*fade)
                                b = int(b*fade)
                                new_rgb = (r << 16) | (g << 8) | b
                                current_colors[beam_index_list[update_index]] = new_rgb
                                strip.set_pixel_rgb(beam_index_list[update_index], new_rgb)
                        strip.show()
                strip.show()
                # Turn off residual pixels after reaching end 
                for index in beam_index_list:
                    strip.set_pixel_rgb(index, 0x000000)
                strip.show()
        # Raise error if an unrecognized display mode is requested
        else:
            raise ValueError("mode must be 'progressive' (default) or 'full' of 'planar' or 'fade'")
