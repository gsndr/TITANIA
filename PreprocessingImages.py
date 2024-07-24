import os
import numpy as np
from skimage import io
import tifffile as tiff
from tiler import Tiler, Merger


class PreprocessingImages():
    def __init__(self, tilesSizeImage, tileSizeMaks, resizeChannel, listToremove, tiles=1, scale=0):
        self.tiles=tiles
        self.tileSize=tilesSizeImage
        self.tileMask=tileSizeMaks
        self.resizeChannel=resizeChannel
        self.toremove=listToremove
        self.scale=scale
        self.scaler=None



    def images(self, pathImages, train, pathTest=None):
        newPath=pathImages+'Numpy/'
        self.tiffToNumpy(pathImages+'Tiff/', newPath)
        self.check_nan(newPath, newPath)
        if self.resizeChannel:
            self.removeBand(newPath,newPath, self.toremove)
        if self.tiles:
            self.get_tiles(newPath,pathImages+'Tiles/', self.tileSize)
        '''
        if self.scale ==1:
            print("scale 10.000")
            self.scaleDataDouble(pathImages+'Tiles/',pathImages+'Tiles/')
        elif self.scale == 3:
            print("Scale Min Max")
            if train:
                self.min_max(pathImages + 'Tiles/', pathImages + 'Tiles/', train=1)
            else:
                self.min_max(pathImages + 'Tiles/', pathImages + 'Tiles/', train=0)
        elif self.scale == 4:
            print("scale 10.000 and 100")
            self.scaleDataDouble(pathImages + 'Tiles/', pathImages + 'Tiles/')
        '''
                
                
    def masks(self, path, ds):
        newPath = path + 'Numpy/'
        self.tiffToNumpy(path+'Tiff/',newPath)
        self.check_nan(newPath, newPath)
        self.changeValueMask(newPath, newPath, ds)
        if self.tiles:
            self.get_tiles(newPath, path+'Tiles/', self.tileMask)


    def get_tiles(self,in_path, outpath, tile_shape):
        self.deleteTiles(outpath)
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                image = np.load(os.path.join(in_path, file))
                # Setup tiling parameters
                print(image.shape)
                tiler = Tiler(data_shape=image.shape,
                              tile_shape=tile_shape, mode='irregular',
                              channel_dimension=None)
                for tile_id, tile in tiler(image):
                    print(f'Tile {tile_id} out of {len(tiler)} tiles.')
                    savefile = os.path.splitext(file)[0] + '_' + str(tile_id)
                    np.save(os.path.join(outpath, savefile), tile)



    def tiffToNumpy(self, in_path, out_path):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                imgarr = io.imread(os.path.join(root, file))
                #imgarr=imgarr[:,:, 13]
                #print(imgarr.dtype)
                #print(imgarr.shape)
                #print(imgarr[0])
                np.save(os.path.join(out_path, file)+'.npy', imgarr)


    def check_nan(self, in_path, out_path):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr = np.load(os.path.join(in_path, file))
                #print(arr.shape)
                #print("Test array for NaN...",np.isnan(arr))
                np.nan_to_num(arr, copy=False)
                #print("Test array for NaN...", np.isnan(arr))

                # print(imgarr.dtype)
                # print(imgarr[0])
                np.save(os.path.join(out_path, file), arr)

    def NumpytoTiff(self, in_path, out_path):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:

                arr=np.load(os.path.join(in_path, file))
                #print(arr.dtype)

                #arr=np.asarray(arr,dtype='uint8')
                arr = np.asarray(arr)
                arr=arr*255
                print(type(file))
                tiff.imwrite( os.path.join(out_path, file)+'.tiff', arr)

    def removeBand(self, in_path, out_path, bands: list):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr=np.load(os.path.join(in_path, file))

                arr = np.delete(arr, bands, axis=2)
                print(arr.shape)
                
                np.save(os.path.join(out_path, file), arr)

    def scaleData(self,in_path, out_path):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr=np.load(os.path.join(in_path, file))
                arr=arr/10000
                np.save(os.path.join(out_path, file), arr)


    def scaleDataDouble(self,in_path, out_path):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr=np.load(os.path.join(in_path, file))
                if arr.shape[2] < 13:
                    print("divide 10000")
                    arr=arr/10000

                else:
                    for i in range(0,11):
                        arr[:,:,i]=arr[:,:,i]/10000
                    #arr[:,:,0:11]=arr[:,:,0:11]/10000
                    for i in range(11,14):
                        arr[:,:,i]=arr[:,:,i]/100
                    #arr[:, :, 11:13]= arr[:, :, 11:13]/100
                np.save(os.path.join(out_path, file), arr)

    def changeValueMask(self, in_path,out_path, ds):

        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr = np.load(os.path.join(in_path, file))
                #print(np.unique(arr))
                if ds=="FIRE":
                    arr[arr <= 36] = 0
                    arr[arr > 36] = 1
                else:
                    arr[arr == 255] = 1
                    #arr[arr == dictChange[1]] = 1
                arr[arr == np.nan] = 0

                print(np.unique(arr))
                np.save(os.path.join(out_path, file) , arr)





        
    def deleteTiles(self, directory):
        for root, _, files in os.walk(directory):
            files.sort()
            for f in files:
                os.remove(os.path.join(directory, f))






    def min_max(self,in_path,out_path, train=0):
        from sklearn.preprocessing import MinMaxScaler
        import os
        import numpy as np
        list_of_images=[]

        if train:
            for root, _, files in os.walk(in_path):
                files.sort()

            for file in files:
                arr = np.load(os.path.join(in_path, file))
                # Convert image to NumPy array
                img_array = np.array(arr)

                # Flatten the image array
                flat_img_array = img_array.flatten()
                list_of_images.append(flat_img_array )



            # Convert the list of images to a 2D array
                # Concatenate all image data into a single array
            images_array = np.concatenate(list_of_images).reshape(-1, 1)
            print(images_array.shape)



            # Initialize the MinMaxScaler
            scaler = MinMaxScaler()

            # Fit the scaler to the data and transform the data
            self.scaler = scaler.fit(images_array)

        from PIL import Image
        for root, _, files in os.walk(in_path):
            files.sort()

        for file in files:
            arr = np.load(os.path.join(in_path, file))
            print(arr.shape)
            flat_img_array = arr.flatten().reshape(-1, 1)

            # Apply Min-Max scaling
            scaled_img_array = self.scaler.transform(flat_img_array).reshape(arr.shape)
            print(scaled_img_array.shape)

            np.save(os.path.join(out_path, file), scaled_img_array)










