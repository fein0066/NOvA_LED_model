import apa102 #<-- this is the driver for the LEDs
import time
import h5py
import numpy as np
from enum import Enum

from led_display import Event
from led_display import EventType

# Input simulation file
f=h5py.File("postprocessed_miniprod6_1_eventonly_small.h5")
  
# Information extracted from input file
shape=f['pixelmap_shape']
coords=f['pixelmap_coordinates']
values=f['pixelmap_values']
evt_index=f['pixelmap_compressed_index']
evt_class=f['event_class']

# Set the number of LED boards (individual modules, 2 horiz rows 2 vert rows)
num_boards = 32
# Multiply boards by LEDs per board to produce total number of LEDs
num_led = 32*num_boards

# Initialize LED hardware
strip = apa102.APA102(num_led=num_led, global_brightness = 10, spi_bus = 0, bus_speed_hz = int(2.5*10**6)) # Bus speed affects the time between the lighting of sequential LEDs; bus speed increases, time decreases

SIM_SHAPE = (100, 80) # Simulation is unpacked into arrays of this shape



# This list of indices is merely a preselected subset of events in the provided file that are known to "look good".
# There is no need to restrict the display to these, it will work with any event.
event_index_list = [4,5,16,18,26,30,34,36,38,48,49,52,53,58,61,66,72,78,83,85,88,97,99,103,109,110]

# Initialize timer
event_loop_start = time.time()

# Loop over specified event indices
for idx in event_index_list:
    # Initialize "full" horiz and vert arrays (containing zeros wehre no hits exist)
    horiz_full = np.zeros(SIM_SHAPE)
    vert_full = np.zeros(SIM_SHAPE)
    # Get coords, values, and event type
    start, stop = evt_index[idx]
    event_type = evt_class[idx]
    event_values = values[start:stop]
    event_coords = coords[start:stop]
    # Populate full arrays with data from one event from input file
    for (y,x), (value_vert, value_horiz) in zip(event_coords, event_values):
        if value_horiz != 0:
            horiz_full[y,x] = value_horiz
        if value_vert != 0:
            vert_full[y,x] = value_vert


    # The next several lines of code are methods acting on out "Event" object.
    # This file is intended to explain how exactly these methods must be used to
    # produce a displayable event, and what options are available for customization
    # of the event display. It is CRITICAL that this chain of methods be respected
    # and not altered. Each method plays an important role in the transformation
    # of data from the provided file into a format capable of being displayed
    # by the physical LED model. Plenty of customization options for the display
    # exist and are explained below, but please be aware that any haphazard
    # attempts to further customize beyond the preordained chain of methods
    # are likely to require a fair amount of tinkering within the "Event" class
    # declaration file led_display.py and not here.


    # Create "Event" object, corresponds to one event from the input file.
    # Requires an unpacked 2D array for the horiz data and vert data (unpacking)
    # process shown above, and the corresponding event type, which the EventType
    # function sitting within the argument of this constructor converts from a
    # given integer to a specified Enum containing the name of the interaction
    # (e.g. NUMU_QE, NUE_DIS, NC, COSMIC, etc.)
    event = Event(horiz_full, vert_full, EventType(event_type))

    # Crops horiz and vert arrays such that they capture only a 3D rectangular
    # region surrounding the active pixels of the event, effectively clipping
    # any surrounding dead pixels. That is, the "Event" object becomes a subset
    # of what it originally was, with the inactive regions of the event removed.
    # The arguments "cutoff" and "threshold" allow for fine control over what pixels
    # are deemed inactive. The argument "threshold" is the fraction of the value
    # of the most energetic pixel of the event below which pixels are removed.
    # In this case, we are saying that any pixel containing less than 1% of the
    # energy value of the maximum pixel is to be removed. If the argument "cutoff"
    # is True, then the effect of "threshold" just explained will take effect.
    # If "cutoff" is False, then only pixels with exactly 0 energy deposited will
    # be truncated from the active region of the event, regardless the unput value
    # provided to "threshold". The behavior of this function in cases where 
    # "threshold" is set to 0 is untested, and is not recommended.
    event.crop_event(cutoff=True, threshold=0.01)

    # Before proceeeding with the explaination of this method, please note that
    # .stretch_event() is the ONLY method in this chain that is not critical
    # to the event display procedure, and thus it is the ONLY method that can
    # be removed from the process. For this reason, it is commented out in the
    # default state of this file to show that it is unnecessary, but it can be
    # uncommented at the user's wish. It exists as a customization option only.
    # Also note that if .stretch_event() is included in the chain of methods,
    # it must lie here, between .crop_event() and .pad_event(), otherwise it
    # will not function properly.
    # Two main behaviors: model-sized stretching and random stretching:
    #   - Model-sized stretching occurs when "enable_random" is set to false.
    #   In this mode, the 3D rectangular region bounding the active pixels
    #   of the event is stretch to occupy exactly the same proportions as the
    #   physical LED model. That is, events displayed in this mode will take
    #   up as much space as possible within the LED model while still being
    #   entirely visible. Note that this will likely cause distortion, but
    #   measures are taken to ensure that the total energy of the event is
    #   conserved after the stretch.
    #   - Random stretching occurs when "enable_random" is set to true. In this
    #   mode, the 3D rectangular region bounding the active pixels of the event
    #   is stretched to a random size smaller than the full LED model, subject
    #   to the constraints set by the arguments "min_random" and "max_random".
    #   These values set the minimum and maximum fractions of the total model
    #   size that the randomizer will allow events to be sized within. Note
    #   that the randomizer produces a different stretching constant for each
    #   of the 3 dimensions of the rectangular region contianing the event, and
    #   all of these stretching constants are subject to the same constraints.
    #   Note that this will likely cause distortion, but measures are taken to
    #   ensure that the total energy of the event is conserved after the
    #   stretch.
#    event.stretch_event(enable_random=False, min_random=0.5, max_random=1)

    # Adds padding zeros around the active region of the event until the result
    # has a shape proportional to the physical LED model. By default, the event
    # is placed at the center of the leftmost edge of the LED model. The arguments
    # allow for movement of the event away from this default position. Note
    # that all of these areguments take integer values corresponding to the
    # number of pixels moved along each axis. Also note that this is the number
    # of pixels within the data file, not the number of LEDs on the physical
    # model. The constant "ARRAY_SHAPE" in led_display.py indicates the size
    # of these pixels. By default, the long side of the model is 160 pixels
    # and the two short sides of the model is 80 pixels.
    # As written, the code underlying this method allows offsets to be greater
    # than the size of the model. When this happens, the code detects when there
    # is no longer any overlap between the active region of the event and the
    # model, at which point it will not light any LEDs. Controls for confining
    # events to the bounds of the LED model can be added to this file or similar
    # by conditioning the values of the offset arguments on "event.horiz.size"
    # or "event.vert.size", etc.
    event.pad_event(x_offset_horiz=0, x_offset_vert=0, y_offset=0)

    # Compresses the event-sized arrays within the "Event" object down to
    # super-pixels of exactly the same number and shape as the arrangement of
    # the LEDs in the physical model. There are three "mode" options for
    # how the super-pixels are populated:
    #   - "sum" - Add the values of all pixels within a super-pixel and assign
    #   that value to the super-pixel
    #   - "max" - Take the value of the maximum pixel within a super-pixel and
    #   assign that value to the super-pixel
    #   "mean" - Take the mean value of all pixels within a super-pixel and
    #   assign that value to the super-pixel
    event.compress_to_led_grid(mode="sum")

    # Converts all energy values within the event arrays to hexadecimal color
    # values logarithmically according to the specified color map. The argument
    # "threshold" encodes the fraction of the maximum value OF THE ENTIRE
    # ENERGY SCALE ACROSS ALL EVENTS below which LEDs will not be enabled.
    # This maximum value is encoded in the constant "vmax" in led_display.py.
    # Note that te color scale is logarithmic, so "threshold" must be greater
    # than zero as not to cause an error.
    # The argument "cmap_name" takes the names of any of the default cmaps
    # shown in the matplotlib documentation easily found on the internet.
    # Known good-looking cmaps include "jet", "inferno", and "rainbow".
    event.values_to_hex(cmap_name="jet", threshold=0.01)

    # Assigns a new attribute "event.led_array" that is easily readable by
    # the LED driver. No human interaction required at this step, just ensure
    # it is included to convert LED data to the proper format.
    event.led_output()

    # Enables LEDs specified by the argument "strip" according to the lighting
    # data stored in the "Event" object's attribute "event.led_array". Must be
    # told the total number of LEDs in the strip via the argument "num_led".
    # There are 4 currently operative inputs for the argument "mode", each
    # of which causes the LEDs to display the event in a different pattern.
    # - "full" - All LEDs simultaneously light, displaying the entire event
    #   in one instant
    # - "progressive" - LEDs light as an invisible plane passes through the
    #   model and reaches their location. All LEDs remain lit after the invisible
    #   plane has passed by their location, causing the entire event to be lit
    #   at once at the end. Plane passes through horizontally for beam events
    #   and vertically for cosmic events.
    # - "fade" - Same functionality as "progressive", but instead of all LEDs
    #   remaining lit until the entire event is diaplayed, LEDs start dimming
    #   after the invisible plane has passed by their location, creating an
    #   effect of active pixels appearing to move through the model. The
    #   invisible plane always passes through the model horizontally in this
    #   mode, regardless of event type.
    # - "planar" - Same functionality as "fade", but instead of the fading
    #   effect, LEDs turn off after the invisible plane has passed by their
    #   location, causing views of only slices of the event at any given moment.
    event.display(strip, mode="progressive", color_enabled=True, num_led=num_led)

    time.sleep(2)
    for index in range(num_led):
        strip.set_pixel_rgb(index, 0x000000)
    strip.show()
