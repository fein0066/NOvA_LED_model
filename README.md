# NOvA_LED_model
Codebase for the operation of tabletop LED model of the NOvA far detector



## Hardware:
The critical pieces of hardware making up the LED model are a strip of individually addressable APA102 RGB LEDs and a Raspberry Pi to control them. The LED strip is divided into subsets of 64 LEDs, each subset corresponding to one module of the model. Modules are connected in series, and the first module is connected to the Raspberry Pi.

This codebase was tested on a model containing 16 LED modules, but is in principle designed to acomodate any number of modules.



## Software:
The files contained in this repository contain sufficient code infrastructure to operate the LED model. Neutrino event data of the same form as the provided `.h5` file is required as an input. The file `led_display.py` defines a custom `Event` class that allows for data from the input file to be converted into a lighting pattern displayed on the physical model. This class is to be imported to a script that reads in a data file, constructs an `Event` object, then calls a sequence of methods to process and display the data. Appropriate data processing is displayed and thoroughly explained within `example_loop.py`. Options for customization of the display are included. Similar scripts can be written to produce scripted display routines.



## Instructions for use of `example_loop.py` example script:
* Ensure hardware is properly set up. Code from this repository is executed on the Raspberry Pi, which must be able to properly control the LEDs
* Ensure the necessary files are present on the Raspberry Pi:
    - Input data file: `postprocessed_miniprod6_1_eventonly_small.h5`
    - `led_display.py` for event processing controls
    - `.py` file(s) containing any display routines to be run; repository-included file `example_loop.py` works by default as a well-documented example
* Execute lighting routine file. The command to run the example is `python example_loop.py`