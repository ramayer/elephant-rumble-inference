import dataclasses
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn.functional
from . import audio_file_processor as afp

class AudioFileVisualizer:
    def __init__(self):
        pass

    def interpolate_1D_tensor(self, input_tensor, target_length):
        # kinda crazy, but:
        # https://stackoverflow.com/questions/73928655/resizing-a-vector-by-interpolation
        z = input_tensor[None,None,:]
        z2 = torch.nn.functional.interpolate(z,target_length)[0][0]
        return z2
    
    def make_similarity_discrete(self, similarity, dissimilarity):
        sim = similarity - dissimilarity
        sim /= sim.abs().max()
        g4 = sim * 0
        r4 = sim * 0
        g4[sim > -0.05] = 1 # Note this threshold can depend on each training run of the model.
        r4[sim <  0.05] = 1 # the loss function only cares which is greater, not by how much.
        g4[sim > 0] = 1
        r4[sim < 0] = 1
        return (g4,r4)

    def add_annotation_boxes(self,labels,patch_start,patch_end,axarr,offset=0.2,only=None,color=(0.0, 1.0, 1.0)):
        for row in labels:
            bt,et,lf,hf,dur,fn,tags,notes,tag1,tag2,score,raven_file = dataclasses.astuple(row)
            if et < patch_start:
                continue
            if bt > patch_end:
                continue
            if only is not None and only != bt:
                continue
            rect = patches.Rectangle((bt - patch_start -offset, lf-5), (et-bt+offset*2), (hf-lf+10), linewidth=3, edgecolor=(0,0,0), facecolor='none')
            axarr.add_patch(rect)
            rect = patches.Rectangle((bt - patch_start -offset, lf-5), (et-bt+offset*2), (hf-lf+10), linewidth=1, edgecolor=color, facecolor='none')
            axarr.add_patch(rect)


    def visualize_audio_file_fragment(self,
                          title,
                          save_file,
                          audio_file,
                          similarity_scoresz,
                          dissimilarity_scoresz,
                          audio_file_processor:afp.AudioFileProcessor,
                          start_time=0,
                          end_time=60*6,
                          height=1280/100,
                          width=1920/100,
                          make_discrete=False,
                          labels=[]
                          ):
        import time
        t0 = time.time()
        start_index = audio_file_processor.time_to_score_index(start_time)
        end_index = audio_file_processor.time_to_score_index(end_time)
        similarity = similarity_scoresz[start_index:end_index].clone()
        dissimilarity = dissimilarity_scoresz[start_index:end_index].clone()

        n_fft = 1024
        hop_length = n_fft//4
        duration = end_time-start_time
        audio,sr = librosa.load(audio_file,sr=1000,offset=start_time,duration=end_time-start_time)
        actual_duration = audio.shape[0] / sr
        print(f"  loaded audio in {time.time()-t0}")
        print(f"  duration","intended=",duration,"actual=",actual_duration)

        spec  = librosa.stft(audio,n_fft=n_fft,win_length=n_fft,hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        s_db  = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        np_spectral_power = np.abs(spec)**2

        print(f"  did stft in {time.time()-t0}")

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
            clip_level = np.percentile(np_spectral_power, 99.9)
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
        print(f"  coloring at {time.time()-t0}")

        stretched_similarity = self.interpolate_1D_tensor(similarity,spec.shape[1])
        stretched_dissimilarity = self.interpolate_1D_tensor(dissimilarity,spec.shape[1])

        if make_discrete:
             s,d = self.make_similarity_discrete(stretched_similarity,stretched_dissimilarity)
             stretched_similarity,stretched_dissimilarity = s,d

        ## An overcomplex color map
        nearness = stretched_similarity
        farness  = stretched_dissimilarity

        sim = nearness-farness
        sim /= sim.abs().max()
        sim = sim.numpy()
        
        redness = -sim * 8 + 1
        redness[redness>1] = 1
        redness[redness<0] = 0

        greenness = sim * 8 + 1
        greenness[greenness>1] = 1
        greenness[greenness<0] = 0
        
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
        print(f"  plotting at {time.time()-t0}")

        plt.ioff()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(width, height),
                                       gridspec_kw={'height_ratios': [3, 1]}
                                       )

        #ts.show(s_db_rgb[300:0:-1,:],figsize=(48, 5))
        #plt.figure(figsize=(199, 3))
        librosa.display.specshow(s_db_rgb[:,:], sr=sr, n_fft=n_fft, hop_length=hop_length, x_axis='time', y_axis='log', y_coords=freqs, ax=ax1)
        plt.gca().set_xticks(np.arange(0, duration, 30))
        #add_annotation_boxes(labels,start_time,duration,plt.gca(),offset=.5)
        print(f"  specshow done at {time.time()-t0}")

        # local_rfw.add_annotation_boxes(labels,start_time,duration,ax1,offset=.5,color=(0,1,0))
        # negative_lables = local_rfw.get_negative_labels(labels)
        # local_rfw.add_annotation_boxes(negative_lables,start_time,duration,ax1,offset=.5,color=(1,0,0))
        self.add_annotation_boxes(labels,start_time,end_time,ax1,offset=0.5,color=(0,0,1))

        #print("make sure similarity shape is compatible",s_db_rgb.shape, stretched_similarity.shape)
        fairseq_time = [i*duration/similarity.shape[0] for i in range(similarity.shape[0])]
        ax2.plot(fairseq_time,similarity_scoresz[start_index:end_index], color='tab:green')
        ax2.plot(fairseq_time,dissimilarity_scoresz[start_index:end_index], color='tab:red')
        ax1.set_xlim(0, duration)
        ax2.set_xlim(0, duration)

        # add a title with some room
        hour,minute,second = int(start_time//60//60),int(start_time//60)%60,start_time%60
        displaytime = f"{hour:02}:{minute:02}"

        plt.subplots_adjust(top=0.93,left=0)
        fig.suptitle(f"{title}", fontsize=16,  ha='left', x=0)
        print(f"  saving at {time.time()-t0}")

        if matplotlib_fixed_issue_26150:=True:
            #fig.tight_layout()
            #print('fig.subplotpairs',fig.subplotpars)
            # Prettier but buggy
            # https://github.com/matplotlib/matplotlib/issues/26150
            plt.savefig(save_file, bbox_inches='tight', pad_inches=0.02)
        else:
            plt.savefig(save_file)
        plt.close()
        plt.close('all')
        print(f"  visualizations saved to {save_file} at {time.time()-t0}")
        #plt.show()


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