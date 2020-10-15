 def extract_data(no_of_set = 1):
   if(no_of_set > 6 or no_of_set < 1):
     print('No of sets should be not be less that 1 and greater that 6')
     return
   for i in range(1,no_of_set+1):
     start = time.time()
     if (os.path.isdir("data_"+str(i))):
         print (f'Imagesof set {i} already downloaded...')
         
     else:
           archive = zipfile.ZipFile(f'/content/gdrive/My Drive/Mask_Rcnn/Dataset/data_part{str(i)}.zip')
           archive.extractall()
         end = time.time()
       print(f"data set extraction took {round(end-start,2)}s") 