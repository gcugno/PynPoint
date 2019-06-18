from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress, memory_frames, locate_star
from pynpoint.util.image import shift_image, rotate

from scipy.ndimage.filters import generic_filter, gaussian_filter
from PyAstronomy.pyasl import crosscorrRV
import numpy as np
import time, copy, sys
from sklearn.decomposition import PCA


from matplotlib import pyplot as plt
#import math
#from scipy.interpolate import interp1d
#import warnings


class SelectWavelengthRangeModule(ProcessingModule):
    """
    Module to eliminate regions of the spectrum.
    """
    
    def __init__(self,
                 range_f,
                 range_i = (1.92854,2.47171),
                 name_in = "Select_range",
                 image_in_tag = "initial_spectrum",
                 image_out_tag = "spectrum_selected",
                 wv_out_tag = "wavelengths"):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_wv: standard deviation of gaussian kernel (in arcseconds)
            :type range_wv: float
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(SelectWavelengthRangeModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_wv_out_port = self.add_output_port(wv_out_tag)
    
        self.m_range_f = range_f
        self.m_range_i = range_i
    
    
    def run(self):
        """
        Run method of the module. Convolves the images with a Gaussian kernel.
            
        :return: None
        """
        
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        
        #self.m_wv_out_port.del_all_data()
        #self.m_wv_out_port.del_all_attributes()
        
        nframes = self.m_image_in_port.get_attribute("NFRAMES")

        spectrum_arr = np.linspace(self.m_range_i[0], self.m_range_i[-1], nframes[0])
        n_before = 0
        while spectrum_arr[n_before]<self.m_range_f[0]:
            n_before+=1
        
        n_after = nframes[0]-1
        while spectrum_arr[n_after]>self.m_range_f[1]:
            n_after-=1
        
        start_time = time.time()
        for i, nframes_i in enumerate(nframes):
            progress(i, len(nframes), 'Running SelectWavelengthRangeModule...', start_time)
            frames_i = self.m_image_in_port[i*nframes_i+n_before:i*nframes_i+n_after,:,:]
            self.m_image_out_port.append(frames_i)
        
        sys.stdout.write('Running SelectWavelengthRangeModule... [DONE]\n')
        sys.stdout.flush()

        nframes_final = np.ones(len(nframes), dtype=int)*len(spectrum_arr[n_before: n_after])

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_attribute("NFRAMES",nframes_final, False)
        self.m_image_out_port.add_history("Select Spectrum", "lambda range = "+str(self.m_range_f))
        self.m_image_out_port.close_port()

        self.m_wv_out_port.set_all(spectrum_arr[n_before: n_after])
        self.m_wv_out_port.add_history("Select Spectrum", "lambda range = "+str(self.m_range_f))
        self.m_wv_out_port.close_port()



class NanSigmaFilterModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 name_in = "Substitute_NaNs",
                 image_in_tag = "im_arr",
                 image_out_tag = "im_arr_Nan"):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(NanSigmaFilterModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
    
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        
        def clean_NaNs(image):
            size=np.shape(image)[0]
            image_filter = generic_filter(image, np.nanmedian, size=3)
            for i in range(size):
                for j in range(size):
                    if not np.isfinite(image[i,j]):
                        image[i,j]=image_filter[i,j]
            return image
        
        
        
        
        self.apply_function_to_images(clean_NaNs,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running NanSigmaFilterModule...")
        
        

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("NaN removed", "sub with mean ")
        self.m_image_out_port.close_port()




class IFUAlignCubesModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 precision = 0.02,
                 shift_all_in_tag= "centering_all",
                 shift_cube_in_tag = "centering_cubes",
                 interpolation="spline",
                 name_in="shift_no_center",
                 image_in_tag="spectrum_NaN_small",
                 image_out_tag="cubes_aligned"):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_wv: standard deviation of gaussian kernel (in arcseconds)
            :type range_wv: float
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(IFUAlignCubesModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_shift_all_in_port = self.add_input_port(shift_all_in_tag)
        self.m_shift_cube_in_port = self.add_input_port(shift_cube_in_tag)

        self.m_image_out_port = self.add_output_port(image_out_tag)
    
        self.m_interpolation = interpolation
        self.m_precision = precision
    
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        
        def _image_shift(image, shift_yx, interpolation):
            return shift_image(image, shift_yx, interpolation)
        
        def _clean_shifts(shift_arr, precision, k):
            x = range(len(shift_arr))
            p = np.polyfit(x,shift_arr,2)
            y = np.polyval(p,x)

            std = np.std(shift_arr-y)
            max_ = np.max(shift_arr-y)
            mean = np.mean(shift_arr-y)
            shift_temp = copy.copy(shift_arr)
            
            #count=0
            while max_>precision:
                filter_shift = generic_filter(shift_temp, np.mean,footprint=np.array([1,0,1]))
    
                for i in range(len(x)):
                    if (shift_temp[i]-y[i] > mean+std) or (shift_temp[i]-y[i] < mean-std):
                        shift_temp[i]=filter_shift[i]

                p = np.polyfit(x,shift_temp,2)
                y = np.polyval(p,x)
                std = np.std(shift_temp[3:-3]-y[3:-3])
                max_ = np.max(shift_temp[3:-3]-y[3:-3])
                mean = np.mean(shift_temp[3:-3]-y[3:-3])
        #count+=1
                    #if count>25:
                    #print (k)
                    #print (np.argmax(shift_temp[3:-3]-y[3:-3]),max_, mean, std)
                    #if k == 20 and count==27:
                    #fig, ax = plt.subplots(1,1)
                    #ax.plot(shift_temp)
                    #ax.plot(filter_shift)
                    #fig.savefig('/scratch/user/gcugno/SINFONI/Results/Reduction_steps/shifts.png')
                            
            
            return shift_temp
        
        nframes = self.m_image_in_port.get_attribute("NFRAMES")
        size = self.m_image_in_port.get_shape()[1]
        
        shift_all_xy = -1.*self.m_shift_all_in_port[:, [0, 2]]
        shift_cubes_xy = -1.*self.m_shift_cube_in_port[:, [0, 2]]
        
        
        start_time = time.time()
        for i, nframes_i in enumerate(nframes):
            progress(i, len(nframes), 'Running IFUAlignCubesModule...', start_time)
            shift_xy_i = shift_all_xy[i*nframes_i:(i+1)*nframes_i]-shift_cubes_xy[i]
            shift_y_i = _clean_shifts(shift_xy_i[:, 1], self.m_precision, i)
            shift_x_i = _clean_shifts(shift_xy_i[:, 0],self.m_precision, i)
            for j in range(nframes_i):
                im = _image_shift(self.m_image_in_port[i*nframes_i+j],[shift_y_i[j], shift_x_i[j]],self.m_interpolation)
                self.m_image_out_port.append(im.reshape(1,size, size))
        
        sys.stdout.write('Running IFUAlignCubesModule... [DONE]\n')
        sys.stdout.flush()
        

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Align", "cube")
        self.m_image_out_port.close_port()




class IFUStellarSpectrumModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 name_in = "Select_range",
                 image_in_tag = "initial_spectrum",
                 wv_in_tag = "wavelengths",
                 spectrum_out_tag = "spectrum_selected",
                 wv_out_tag = "wavelengths",
                 num_pix = 20,
                 std_max=0.2):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_wv: standard deviation of gaussian kernel (in arcseconds)
            :type range_wv: float
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(IFUStellarSpectrumModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_spectrum_out_port = self.add_output_port(spectrum_out_tag)
    
        self.m_num_pix = num_pix
        self.m_std = std_max
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        self.m_spectrum_out_port.del_all_data()
        self.m_spectrum_out_port.del_all_attributes()
        
        def collapse_frames(images):
            return np.mean(images, axis=0)
        
        def find_brightest_pixels(image, num_pix):
            im_new = np.copy(image)
            Count = 0
            max_pos = []
            while Count<num_pix:
                max_pos_i = np.unravel_index(np.argmax(im_new, axis=None), im_new.shape)
                max_pos.append(max_pos_i)
                Count+=1
                im_new[max_pos_i]=0
            return max_pos
        
        def _normalize_spectra(spectra_init, num_pix):
            spectra2 = np.zeros_like(spectra_init)
            for p in range(num_pix):
                spectra2[p,:] = spectra_init[p,:]/spectra_init[p,0]#np.mean(spectra_init[p,:])
            return spectra2
                
        def _find_outliers(spectra_init, std):
            spectra2 = copy.copy(spectra_init)
            spectrum_f = np.zeros(nspectrum[0])
            for m in range(nspectrum[0]):
                new_arr=spectra2[:,m]
                while np.std(new_arr)>std:
                    m_i = np.mean(new_arr)
                    d_i = np.argmax(np.abs(new_arr-m_i))
                    new_arr = np.delete(new_arr,d_i)
                spectrum_f[m] = np.mean(new_arr)
            return spectrum_f
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUStellarSpectrumModule...', start_time)
            frames_i = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            cube_collapsed = collapse_frames(frames_i)
            pixels = find_brightest_pixels(cube_collapsed,10)
            
            pix_spectrum = np.zeros((self.m_num_pix, nspectrum_i))
            for k in range(self.m_num_pix):
                pix_spectrum[k,:] = frames_i[:,pixels[k][0], pixels[k][1]]
            
            pix_spectrum_norm = _normalize_spectra(pix_spectrum, self.m_num_pix)
            pix_spectrum_norm_outliers = _find_outliers(pix_spectrum_norm, self.m_std)
            
            self.m_spectrum_out_port.append(pix_spectrum_norm_outliers)
        
        sys.stdout.write('Running IFUStellarSpectrumModule... [DONE]\n')
        sys.stdout.flush()
        
        self.m_spectrum_out_port.copy_attributes(self.m_image_in_port)
        self.m_spectrum_out_port.add_history("Stellar Spectrum", "num pixels = "+str(self.m_num_pix))
        self.m_spectrum_out_port.close_port()



class IFUPSFSubtractionModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 name_in = "Select_range",
                 image_in_tag = "initial_spectrum",
                 stellar_spectra_in_tag = "stellar_spectrum",
                 image_out_tag = "PSF_sub",
                 gauss_sigma = 10,
                 sigma=3.,
                 iteration = 2):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_wv: standard deviation of gaussian kernel (in arcseconds)
            :type range_wv: float
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(IFUPSFSubtractionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_stellar_spectra_in_port = self.add_input_port(stellar_spectra_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_gauss_sigma = gauss_sigma
        self.m_sigma = sigma
        self.m_iteration = iteration
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        
        def _find_outliers(spectrum, sigma):
            median = np.median(spectrum)
            std = np.std(spectrum)
            spectrum2 = copy.copy(spectrum)
            for i in range(len(spectrum)):
                if (np.abs(spectrum2[i]-median) > sigma*std) and (i > 3) and (i < len(spectrum2)-3):
                    spectrum2[i] = np.median(spectrum2[i-3:i+3])
                        #generic_filter(spectrum, np.median,size=2)[i]
            return spectrum2
        
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        size = self.m_image_in_port.get_shape()[1]
        
        stellar_spectra_master = self.m_stellar_spectra_in_port.get_all().reshape(len(nspectrum), nspectrum[0])
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUPSFSubtractionModule...', start_time)

            frames_i = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            frames_i_final = np.zeros_like(frames_i)
            for k in range(size):
                for j in range(size):
                    spectrum_temp_orig = frames_i[:,j,k]
                    spectrum_temp = frames_i[:,j,k] / stellar_spectra_master[i]
                    for g in range(self.m_iteration):
                        spectrum_temp = _find_outliers(spectrum_temp, self.m_sigma)
                    spectrum_smooth = gaussian_filter(spectrum_temp ,self.m_gauss_sigma)
                    stellar_proxy = spectrum_smooth * stellar_spectra_master[i]
                    residuals = spectrum_temp_orig - stellar_proxy
                    for g in range(self.m_iteration):
                        residuals = _find_outliers(residuals, self.m_sigma)
                    frames_i_final[:,j,k] = residuals
            
            self.m_image_out_port.append(frames_i_final)
        
            if i == 0:
                #print (np.std(frames_i_sub[:,10,29]))
                #fig, ax = plt.subplots(1,1)
                #ax.plot(frames_i[:,10,29], color='red',label='Original')
                '''ax.plot(frames_i_div[:,10,29], color='b',label='Divided by star master')
                ax.plot(frames_i_smooth[:,10,29], color='green',label='smoothed')
                ax.plot(frames_i_st[:,10,29], color='violet',label='Stellar proxy')
                ax.plot(frames_i_sub[:,10,29], color='lightgrey',label='Residuals')'''
                #ax.plot(frames_i_final[:,10,29], color='k',label='outliers removed')


        #ax.plot(pix_spectrum_norm_outliers[8,:], color='blue',label='5')
        
        #ax .set_ylim(-10,80)
        #ax.legend(loc='best')
        
        #fig.savefig('/Users/Gabo/SINFONI/Beta_Pic/Results/PSF_sub.pdf')
        
        
        sys.stdout.write('Running IFUPSFSubtractionModule... [DONE]\n')
        sys.stdout.flush()
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Stellar Spectrum", "gauss proxy = "+str(self.m_gauss_sigma))
        self.m_image_out_port.close_port()



class FoldingModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 name_in = "Select_range",
                 image_in_tag = "initial_spectrum",
                 image_out_tag = "im_2D"):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_wv: standard deviation of gaussian kernel (in arcseconds)
            :type range_wv: float
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(FoldingModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        size = self.m_image_in_port.get_shape()[1]
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running FoldingModule...', start_time)
            
            folded = np.zeros((nspectrum_i, int(size/2), size))
            frames_i = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            for k in range(int(size/2)):
                folded[:,k,:] = (frames_i[:,2*k,:]+frames_i[:,2*k+1,:])/2.
        
            im_2d = np.zeros((nspectrum_i, int(size*size/2)))
            for j in range(nspectrum_i):
                for k in range(int(size/2.)):
                    im_2d[nspectrum_i-j-1,k*size:(k+1)*size]=folded[j,k,:]
                        
            self.m_image_out_port.append(im_2d.reshape(1,nspectrum_i,int(size*size/2)))
            
            sys.stdout.write('Running FoldingModule... [DONE]\n')
            sys.stdout.flush()

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Folded", "2D image ")
        self.m_image_out_port.close_port()


class UnfoldingModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 name_in = "Select_range",
                 image_in_tag = "initial_spectrum",
                 image_out_tag = "im_2D"):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_wv: standard deviation of gaussian kernel (in arcseconds)
            :type range_wv: float
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(UnfoldingModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        size = int(np.sqrt(self.m_image_in_port.get_shape()[2]*2))
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running UnfoldingModule...', start_time)
            
            folded = np.zeros((nspectrum_i, int(size/2), size))
            im_2D = self.m_image_in_port[i,:,:]
            
            for j in range(nspectrum_i):
                for k in range(int(size/2)):
                    folded[j,k,:] = im_2D[nspectrum_i-j-1,k*size:(k+1)*size]

            frames_i = np.zeros((nspectrum_i,size, size))
            for k in range(int(size/2)):
                frames_i[:,2*k,:] = folded[:,k,:]
                frames_i[:,2*k+1,:] = folded[:,k,:]
        
            self.m_image_out_port.append(frames_i)
            
        sys.stdout.write('Running UnfoldingModule... [DONE]\n')
        sys.stdout.flush()
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Unfolded", "3D image ")
        self.m_image_out_port.close_port()




class IFUResidualsPCAModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 pc_number,
                 name_in = "Select_range",
                 image_in_tag = "im_2D",
                 image_out_tag = "im_2D_PCA",
                 model_out_tag = "PCA_model"):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_wv: standard deviation of gaussian kernel (in arcseconds)
            :type range_wv: float
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(IFUResidualsPCAModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_model_out_port = self.add_output_port(model_out_tag)
    
        self.m_pc_number = pc_number
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        self.m_model_out_port.del_all_data()
        self.m_model_out_port.del_all_attributes()
        
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        nim = self.m_image_in_port.get_shape()[0]
        size = self.m_image_in_port.get_shape()[2]
        
        sys.stdout.write('Running IFUResidualsPCAModule...')
        
        im_1D = []
        for k in range(nim):
            im_2D = self.m_image_in_port[k,:,:]-np.mean(self.m_image_in_port[k,:,:])
            im_1D.append(im_2D.reshape(-1))
        im_1D = np.array(im_1D-np.mean(im_1D, axis=0))
        
        pc_max = self.m_pc_number
        pca_sklearn = PCA(n_components=pc_max, svd_solver="arpack")
        pca_sklearn.fit(im_1D)


        zeros = np.zeros((pca_sklearn.n_components - self.m_pc_number, im_1D.shape[0]))
        pca_rep = np.matmul(pca_sklearn.components_[:self.m_pc_number], im_1D.T)
        pca_rep = np.vstack((pca_rep, zeros)).T
        model = pca_sklearn.inverse_transform(pca_rep)
        im_1D_new = im_1D - model
        
        self.m_image_out_port.set_all(im_1D_new.reshape(len(nspectrum),nspectrum[0],size))
        self.m_model_out_port.set_all(model.reshape(len(nspectrum),nspectrum[0],size))
        

#for k in range(nim):
#fig, ax1 = plt.subplots(1,1,figsize=(20,10))
#ax1.plot(im_1D[k], color='red',label='Original')
#ax1.plot(im_1D_new[k], color='black',label='Original')
#ax1.set_ylim(-1500,2000)
#fig.savefig('/scratch/user/gcugno/SINFONI/Results/Reduction_steps/Images_1D%s.png'%(k+1))
        
        sys.stdout.write('Running IFUResidualsPCAModule... [DONE]\n')
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("PCA", "PC=")

        self.m_model_out_port.copy_attributes(self.m_image_in_port)
        self.m_model_out_port.add_history("PCA", "PC=")
        
        self.m_image_out_port.close_port()
        self.m_model_out_port.close_port()



class CrossCorrelationPreparationModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 name_in = "Select_range",
                 image_in_tag = "initial_spectrum",
                 shift_cubes_in_tag="centering_cubes",
                 image_out_tag = "im_2D",
                 mask_out_tag="mask",
                 data_mask_out_tag = "data_mask"):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_wv: standard deviation of gaussian kernel (in arcseconds)
            :type range_wv: float
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(CrossCorrelationPreparationModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_shift_in_port = self.add_input_port(shift_cubes_in_tag)
        
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_mask_out_port = self.add_output_port(mask_out_tag)
        self.m_data_mask_out_port = self.add_output_port(data_mask_out_tag)
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()
        self.m_mask_out_port.del_all_data()
        self.m_mask_out_port.del_all_attributes()
        self.m_data_mask_out_port.del_all_data()
        self.m_data_mask_out_port.del_all_attributes()
        
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        parang = self.m_image_in_port.get_attribute("PARANG")
        size = self.m_image_in_port.get_shape()[-1]
        index = np.array([0,4,10,16,20])
        
        shift_init = self.m_shift_in_port.get_all()
        shift_x = shift_init[:,0]
        shift_y = shift_init[:,2]
        shift = np.array([shift_y,shift_x]).T
        
        final_cube = []
        
        mask_arr = np.zeros((len(nspectrum), size,size))
        mask_arr_shift = np.zeros_like(mask_arr)
        mask_arr_rot = np.zeros_like(mask_arr)
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running CrossCorrelationPreparationModule...', start_time)

            cube_init = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            
            mask_arr[i] = np.where(cube_init[0]==0, False, True)
            mask_arr_shift[i] = shift_image(mask_arr[i], -shift[i], "spline")
            mask_arr_rot[i] = rotate(mask_arr_shift[i], -parang[index[i]], reshape=False)
    
            cube_shift = np.zeros_like(cube_init)
            cube_rot = np.zeros_like(cube_init)
            for k in range(nspectrum_i):
                cube_shift[k] = shift_image(cube_init[k], -shift[i], "spline")
                cube_rot[k] = rotate(cube_shift[k], -parang[index[i]], reshape=False)

            final_cube.append(cube_rot)
        
        final_cube = np.array(final_cube)
        cube_median = np.nanmean(final_cube, axis=0)
        mask_sum = np.sum(mask_arr_rot, axis=0)
        
        mask_final = np.where(mask_sum>=len(nspectrum)/3, True, False)
        mask_output = np.where(mask_sum>=len(nspectrum)/3, 1, 0)
        
        data_mask = []
        
        for k in range(nspectrum[0]):
            images = np.where(np.abs(mask_arr_rot)>0.1, final_cube[:,k,:,:], np.nan)
            data_mask.append(np.where(mask_final, np.nanmedian(images, axis=0), np.nan))
            
        self.m_image_out_port.set_all(cube_median)
        self.m_mask_out_port.set_all(mask_output)
        self.m_data_mask_out_port.set_all(data_mask)
        
                        
        sys.stdout.write('Running CrossCorrelationPreparationModule... [DONE]\n')
        sys.stdout.flush()
                        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("CC prep", "CC prep")
        
        self.m_mask_out_port.copy_attributes(self.m_image_in_port)
        self.m_mask_out_port.add_history("CC prep", "CC prep")
        
        self.m_data_mask_out_port.copy_attributes(self.m_image_in_port)
        self.m_data_mask_out_port.add_history("CC prep", "CC prep")
        
        self.m_image_out_port.close_port()
        self.m_mask_out_port.close_port()
        self.m_data_mask_out_port.close_port()



















class CrossCorrelationModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 name_in = "CrossCorr",
                 RV = 2500,
                 dRV = 10,
                 data_wv_in_tag = "wavelength_range",
                 model_wv = None,
                 model_abs = None,
                 image_in_tag = "data_mask",
                 mask_in_tag = "mask",
                 snr_map_out_tag = "snr",
                 CC_cube_out_tag = "CC_cube"):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_wv: standard deviation of gaussian kernel (in arcseconds)
            :type range_wv: float
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(CrossCorrelationModule, self).__init__(name_in)
        
        self.m_data_wv_in_port = self.add_input_port(data_wv_in_tag)
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_mask_in_port = self.add_input_port(mask_in_tag)
        
        self.m_snr_map_out_port = self.add_output_port(snr_map_out_tag)
        self.m_CC_cube_out_port = self.add_output_port(CC_cube_out_tag)
    
        self.m_RV = RV
        self.m_dRV = dRV
        self.m_wv_model = model_wv
        self.m_abs_model = model_abs
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        self.m_snr_map_out_port.del_all_data()
        self.m_snr_map_out_port.del_all_attributes()
        self.m_CC_cube_out_port.del_all_data()
        self.m_CC_cube_out_port.del_all_attributes()
        
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        size = self.m_image_in_port.get_shape()[-1]
        
        wv = self.m_data_wv_in_port.get_all()
        mask = self.m_mask_in_port.get_all()
        CC_prep = self.m_image_in_port.get_all()
        
        def _CC(wv, spectrum, wv_model, model):
            rv, cc = crosscorrRV(w = wv, f = spectrum, tw = wv_model, tf = model,rvmin=-self.m_RV, rvmax = self.m_RV+self.m_dRV, drv = self.m_dRV,skipedge=200)
            #fit = np.polyfit(rv,cc,1)
            #line = np.polyval(fit,rv)
            #cc_res = cc-line
            #N_arr = np.concatenate((cc_res[:np.argmax(cc_res)-15], cc_res[np.argmax(cc_res)+15:]))
            N_arr = np.concatenate((cc[:np.argmax(cc)-15], cc[np.argmax(cc)+15:]))
            snr = (np.max(cc)-np.mean(N_arr))/np.std(N_arr)
            #return cc,cc_res, rv[np.argmax(cc)], snr
            return cc, rv[np.argmax(cc)], snr
        
        
        CC_cube = np.zeros((int(2*self.m_RV/self.m_dRV+1), size, size))
        rv = np.zeros((size, size))
        snr = np.zeros((size, size))
        
        start_time = time.time()
        for i in range(size):
            for j in range(size):
                progress(i*size+j, size**2, 'Running CrossCorrelationModule...', start_time)
                if mask[i,j]!=0:
                    CC_cube[:,i,j], rv[i,j], snr[i,j] = _CC(wv, CC_prep[:,i,j], self.m_wv_model, self.m_abs_model)
    
    
        self.m_CC_cube_out_port.set_all(CC_cube)
    
        
        
        
        sys.stdout.write('Running CrossCorrelationModule... [DONE]\n')
        sys.stdout.flush()
        
        #self.m_snr_map_out_port.copy_attributes(self.m_image_in_port)
        #self.m_snr_map_out_port.add_history("CC prep", "CC prep")
        
        self.m_CC_cube_out_port.copy_attributes(self.m_image_in_port)
        self.m_CC_cube_out_port.add_history("CC prep", "CC prep")
        
        #self.m_snr_map_out_port.close_port()
        self.m_CC_cube_out_port.close_port()

