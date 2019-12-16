"""
The IRIS model is the function model for this process of base calling.

This model include six sub-models. Four general sub-models are used to transform images into barcode. This 
process includes 1) importing and storing images in a 3D tensor data structure; 2) detecting blobs, transforming
them into bases and calculating their error rate by binomial test under 50% success ratio; 3) connecting called 
bases in the same location from different cycles into the barcode sequences; 4) reformatting the result and 
outputting. The other two particular sub-models to register images among different cycles, and transform blobs 
into bases.
"""
