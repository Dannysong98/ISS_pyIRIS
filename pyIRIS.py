#!/usr/bin/env python3
"""
Description:    This software is used to transform the fluorescent signal of in situ sequencing (ISS) into base
                sequences (barcode sequences), as well as the quality of base and their coordinates in the background
                image.

Author: Hao Yu (yuhao@genomics.cn)
        Yang Zhou (zhouyang@genomics.cn)

ChangeLog:  2019-01-15  r001    First development version
            2019-03-18  r002    Make Simple Blob Detector as the default detector
            2019-03-27  r003    To modify the detecting strategy
            2019-04-08  r004    To modify the quality algorithm
            2019-04-09  r005    To adjust the strategy of function of blob exposing
            2019-04-10  r006    To add the code document to explain primary functions
            2019-04-16  r007    1) To modify the algorithm of registration for getting more true positive blobs
                                2) To modify the algorithm for blob detecting for accuracy
                                3) To modify the parameter set of blob detector for accuracy
            2019-04-24  r008    To modify the algorithm for saturate blob exposing
            2019-04-25  r009    To modify the strategy for blob richly exposing and detection
            2019-05-09  r010    To modify the algorithm for blob exposing
            2019-05-12  r011    To modify the algorithm for blob exposing, twice brighten exposed blobs
            2019-05-13  r012    1) To modify the method of channel merging by abandoning the DAPI dying
                                2) To brighten exposed blobs
            2019-05-16  r013    To modify the merging strategy of channels
            2019-05-17  r014    To modify the strategy for blob richly exposing and detection
            2019-06-27  r015    Only use channel DAPI as the image for registration
            2019-07-09  r016    To modify the algorithm for error rate
            2019-07-12  r017    To modify the algorithm for base cube
            2019-07-12  r018    To modify the algorithm for blob detection
            2019-08-08  r019    New development version, the prototype version
            2019-08-27  r020    To modify the algorithm for base calling, for performance improving
            2019-08-30  r021    To add a function for lightness rectification
            2019-09-26  r022    New public version
            2019-11-25  v1.0    Release version
"""


from sys import (argv, stderr)
from numpy import (array, uint8)

from IRIS import (import_images, detect_signals, connect_barcodes, deal_with_result)


if __name__ == '__main__':
    """
    This is the entry of function in our software, of which, this process can be split three parts, images importing
    module, blob detecting module and base connecting module.
    
    The images importing module is compatible with two types of data, which are generated from the technologies by 
    Ke et al. 2013 and Eng et al. 2017 and 2019, respectively.
    
    Our software control the data importing by two options following the main command, of which, the '--ke' means to 
    process the data belonging the type of R. Ke, and the '--eng' means another type, with respective optimized 
    parameters.
    """
    if len(argv) > 2 and ('--ke' in argv[1] or '--eng' in argv[1] or '--lee' in argv[1] or '--chen' in argv[1]):
        cycle_stack = []
        std_img = array([], dtype=uint8)
        called_base_box_in_one_cycle = {}

        barcode_cube_obj = connect_barcodes.BarcodeCube()

        if argv[1] == '--ke':
            cycle_stack, std_img = import_images.decode_data_Ke(argv[2:])

            for cycle in cycle_stack:
                called_base_box_in_one_cycle = detect_signals.detect_blobs_Ke(cycle)
                barcode_cube_obj.collect_called_bases(called_base_box_in_one_cycle)

            barcode_cube_obj.filter_blobs_list(std_img)

        if argv[1] == '--eng':
            cycle_stack, std_img = import_images.decode_data_Eng(argv[2:])

            for cycle in cycle_stack:
                called_base_box_in_one_cycle = detect_signals.detect_blobs_Eng(cycle)
                barcode_cube_obj.collect_called_bases(called_base_box_in_one_cycle)

        #####################################
        # Handle Weinstein's data (ABANDON) #
        #####################################
        # if argv[1] == '--weinstein':
        #     pass
        #####################################

        if argv[1] == '--lee':
            cycle_stack, std_img = import_images.decode_data_Lee(argv[2:])

            for cycle in cycle_stack:
                called_base_box_in_one_cycle = detect_signals.detect_blobs_Lee(cycle)
                barcode_cube_obj.collect_called_bases(called_base_box_in_one_cycle)

            barcode_cube_obj.filter_blobs_list(std_img)

        if argv[1] == '--chen':
            cycle_stack, std_img = import_images.decode_data_Chen(argv[2:])

            for cycle in cycle_stack:
                called_base_box_in_one_cycle = detect_signals.detect_blobs_Chen(cycle)
                barcode_cube_obj.collect_called_bases(called_base_box_in_one_cycle)

            barcode_cube_obj.filter_blobs_list(std_img)

        #############################################################
        # Start here, different types of data set are to be unified #
        #############################################################
        barcode_cube_obj.calling_adjust()
        #############################################################

        deal_with_result.write_reads_into_file(std_img, barcode_cube_obj.adjusted_bases_cube, len(cycle_stack))

    else:
        print('Invalid image group\nUSAGE:  ' + argv[0] + ' <--ke|--eng|--lee｜--chen> <image group>', file=stderr)
