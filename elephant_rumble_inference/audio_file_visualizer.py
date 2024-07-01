
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
from . import audio_file_processor as afp

class AudioFileVisualizer:
    def __init__(self):
        pass


    def interpolate_and_scale_1D_tensor(self, input_tensor, target_length):
        # kinda crazy, but:
        # https://stackoverflow.com/questions/73928655/resizing-a-vector-by-interpolation
        z=input_tensor[None,None,:]
        z2 = torch.nn.functional.interpolate(z,target_length)[0][0]
        z2 -= np.min(z2.numpy())
        z2 /= np.max(z2.numpy())
        return z2
    
    
    def visualize_audio_file_fragment(self,
                          title,
                          save_file,
                          audio_file,
                          similarity_scoresz,
                          dissimilarity_scoresz,
                          audio_file_processor:afp.AudioFileProcessor,
                          start_time=0,
                          end_time=60*6,
                          height=8,
                          width=24,
                          ):
        
        start_index = audio_file_processor.time_to_score_index(start_time)
        end_index = audio_file_processor.time_to_score_index(end_time)
        similarity = similarity_scoresz[start_index:end_index]
        dissimilarity = dissimilarity_scoresz[start_index:end_index]

        n_fft = 2048
        hop_length = n_fft//4
        duration = end_time-start_time
        audio,sr = librosa.load(audio_file,sr=2000,offset=start_time,duration=end_time-start_time)

        spec  = librosa.stft(audio,n_fft=n_fft,win_length=n_fft,hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        s_db  = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        np_spectral_power = np.abs(spec)**2

        #np_spectral_power = spec.numpy(force=True) # if you used the torchaudio stft
        if try_per_channel_normalization_on_power := True:
            # Qualitatively, dividing by the median power seems better than subtracting the median power.
            median_pwr_per_spectral_band  = np.median(np_spectral_power, axis=1)
            normalized_pwr = np_spectral_power / median_pwr_per_spectral_band[:, None]
            np_spectral_power = normalized_pwr
        else:
            power_in_region_of_interest = np_spectral_power

        if clip_outliers := True:
            #db = db - np.max(db)
            #print("db shape",np_spectral_power.shape)
            noise_floor =  np.percentile(np_spectral_power,0)
            clip_level = np.percentile(np_spectral_power, 99.99)
            db_normalized = np_spectral_power
            db_normalized[np_spectral_power < noise_floor]=noise_floor
            db_normalized[np_spectral_power > clip_level]=clip_level
            #db_normalized = np.clip(db, -60, clip_level)
            np_spectral_power = db_normalized

        s_db  = librosa.power_to_db(np_spectral_power, ref=np.max)

        mx = np.max(s_db)
        mn = np.min(s_db)
        normed = (s_db - mn) / (mx-mn)
        s_db_rgb = np.stack((normed,normed,normed), axis=-1)

        stretched_similarity = self.interpolate_and_scale_1D_tensor(similarity,spec.shape[1])
        stretched_dissimilarity = self.interpolate_and_scale_1D_tensor(dissimilarity,spec.shape[1])

        ## An overcomplex color map
        nearness = stretched_similarity.numpy()
        #nearness[nearness<0] = 0
        nearness -= np.min(nearness)
        nearness /= np.max(nearness)
        nothing_threshold=0
        farness = stretched_dissimilarity.numpy()
        #farness[farness<0] = 0
        farness -= np.min(farness)
        farness /= np.max(farness)
        redness = farness
        greenness = nearness
        blueness = 1-(redness + greenness)
        blueness[blueness<0] = 0
        #blueness = blueness - np.min(blueness)
        #blueness = blueness / np.max(blueness)/2
        s_db_rgb[:,:,0] = s_db_rgb[:,:,0] * (redness)
        s_db_rgb[:,:,1] = s_db_rgb[:,:,1] * (greenness)
        s_db_rgb[:,:,2] = s_db_rgb[:,:,2] * (blueness)

        ##  A one-dimensional colormap not good if the classifier has both a similarity and dissimiliary score
        # rgb_array = self.values_to_rgb(stretched_similarity-stretched_dissimilarity)
        # print("rgb_array.shape",rgb_array.shape)
        # s_db_rgb[:,:,0] = s_db_rgb[:,:,0] * rgb_array[:,0].T
        # s_db_rgb[:,:,1] = s_db_rgb[:,:,1] * rgb_array[:,1].T
        # s_db_rgb[:,:,2] = s_db_rgb[:,:,2] * 0

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(width, height),
                                       gridspec_kw={'height_ratios': [3, 1]}
                                       )

        #ts.show(s_db_rgb[300:0:-1,:],figsize=(48, 5))
        #plt.figure(figsize=(199, 3))
        librosa.display.specshow(s_db_rgb[:,:], sr=sr, n_fft=n_fft, hop_length=hop_length, x_axis='time', y_axis='log', y_coords=freqs, ax=ax1)
        plt.gca().set_xticks(np.arange(0, duration, 30))
        #add_annotation_boxes(labels,start_time,duration,plt.gca(),offset=.5)

        # local_rfw.add_annotation_boxes(labels,start_time,duration,ax1,offset=.5,color=(0,1,0))
        # negative_lables = local_rfw.get_negative_labels(labels)
        # local_rfw.add_annotation_boxes(negative_lables,start_time,duration,ax1,offset=.5,color=(1,0,0))


        #print("make sure similarity shape is compatible",s_db_rgb.shape, stretched_similarity.shape)
        fairseq_time = [i*duration/similarity.shape[0] for i in range(similarity.shape[0])]
        ax2.plot(fairseq_time,similarity, color='tab:green')
        ax2.plot(fairseq_time,dissimilarity, color='tab:red')

        ax1.set_xlim(0, duration)
        ax2.set_xlim(0, duration)

        # add a title with some room
        hour,minute,second = int(start_time//60//60),int(start_time//60)%60,start_time%60
        displaytime = f"{hour:02}:{minute:02}"

        fig.suptitle(f"{title}", fontsize=16,  ha='left', x=0.125)
        plt.subplots_adjust(top=0.93)
        plt.savefig(save_file)
        plt.show()

# AudioFileVisualizer().visualize_audio_file_fragment(
#     f"{audio_file} and scores",
#     '/home/ron/proj/elephantlistening/tmp/aves/test.png',
#     audio_file,
#     scores[:,1],
#     scores[:,0],
#     afp,
#     start_time=0,
#     end_time=60*60
# )