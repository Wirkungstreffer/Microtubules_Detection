
# This function is for read 16-bit images and normalize further save them to 8-bit images 
def read_directory(directory):
    # Define a image list, for further visualization
    array_of_img = []

    for filename in os.listdir(r"./"+ directory):
        # Reading 16-bit images
        tiff_image = cv2.imread(directory + "/" + filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        # Normalize them to 8-bit images
        img = cv2.normalize(tiff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) 

        # Save the transfered images into file "Data_Process/8bit_tiff_file"
        cv2.imwrite("Data_Process/8bit_tiff_file" + "/" + filename, img) 

        # Save images to list "array_of_img" in case for visualization 
        array_of_img.append(img)  
    
    return array_of_img




# This function is for converting image format  
def change_image_format_batch(src_path, tar_path, fmt_in, fmt_out ):
  
  # Check if the input folder exist
  if os.path.exists(src_path)==False:
    raise FileNotFoundError( 'No such file or directory:'+ src_path)
  
  # Create a dictionary of input folder
  img_dict = dict()
  directorys = [ subpath for subpath in os.listdir(src_path) if   os.path.isdir( os.path.join(src_path,subpath) )   ]

  # Reading the names and format of input images
  if len(directorys)==0:
    imgPaths=glob.glob(os.path.join(src_path,'*.'+ fmt_in))
    
    # If output folder is not exist, create such folder
    if os.path.exists(tar_path)==False:
      os.makedirs(tar_path)
    
    # Separate the names and the format of images, store the names
    for imgPath in imgPaths:
      im=Image.open(imgPath)
      _,imgNamePNG=os.path.split(imgPath)
      imgName,PNG=os.path.splitext(imgNamePNG)
      im.save(os.path.join(tar_path,imgName+'.'+ fmt_out))
    return

  # Check the subfolders of input folder in the dictionary
  for subdir in directorys:
    img_dict[subdir]=glob.glob(os.path.join(src_path,subdir,'*.'+ fmt_in))
  
  # If the subfolders is not exist in output folder, create such subfolders
  if os.path.exists(tar_path)==False:
    os.makedirs(tar_path)
  
  # Save the names and changed format into output folder
  for subdir,imgPaths in img_dict.items():
    newLongdir=os.path.join(tar_path,subdir)
    
    # Check if the output folder exist
    if os.path.exists(newLongdir)==False:
      os.makedirs(newLongdir)
    
    # Save the names of images with the wanted format
    for imgPath in imgPaths:
      im=Image.open(imgPath)
      _,imgNamePNG=os.path.split(imgPath)
      imgName,PNG=os.path.splitext(imgNamePNG)
      im.save(os.path.join(tar_path,subdir,imgName+'.'+ fmt_out))


# Define a images loading function
def load_images(image_file_directory):
    # Create a image list
    images = []
    
    # Check if the input folder exist
    if os.path.exists(image_file_directory)==False:
        raise FileNotFoundError( 'No such file or directory:'+ image_file_directory)

    # Reading the images in the folder 
    for directory_path in glob.glob(image_file_directory):
        image_path = glob.glob(os.path.join(directory_path, '*.png'))
        
        # Make sure reading sequence of the images is correctly according to the name sequence of images
        image_path.sort()
        
        # Reading images, dd up into images list
        for i in image_path:     
            images.append(i)
    
    return images


# Define a images loading and padding function, with the input loading data 1200x1200 png images and output 1216x1216 png images
def load_and_padding_images(image_file_directory, channel):

    # Create a image list
    image_set = []

    # Check if the input folder exist
    if os.path.exists(image_file_directory)==False:
        raise FileNotFoundError( 'No such file or directory:'+ image_file_directory)
        
    # Reading the images in the folder 
    for directory_path in glob.glob(image_file_directory):
        img_path = glob.glob(os.path.join(directory_path, '*.png'))

        # Make sure reading sequence of the images is correctly according to the name sequence of images
        img_path.sort()

        # Reading images
        for i in img_path:
            
            if channel == 3:
                # Read the images as RGB mode
                img = cv2.imread(i, cv2.IMREAD_COLOR)
            elif channel == 1:
                # Read the images as binary mode
                img = cv2.imread(i, 0)
            else:
                print("False channel input")

            # Use reflect padding the images into size 1216x1216
            reflect_img = cv2.copyMakeBorder(img,8,8,8,8,cv2.BORDER_REFLECT) 

            # Add up into images list     
            image_set.append(reflect_img)

    # Convert list to array for machine learning processing      
    image_set = np.array(image_set)
    
    return image_set