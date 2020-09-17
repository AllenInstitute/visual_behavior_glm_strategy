import visual_behavior.plotting as vbp
import visual_behavior.data_access.loading as loading
import visual_behavior_glm.GLM_analysis_tools as gat
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import gc
from scipy import ndimage

def plot_regressor_correlation(glm, add_lines=True,save_plot=False):
    '''
        Plots the correlation of the design matrix for this glm object
        
        glm, the session to look at
        add_lines (bool), if True, plots faint lines to devide the correlation matrix
    '''   

    # Look at the discrete event kernels 
    discrete = [x for x in glm.run_params['kernels'] if glm.run_params['kernels'][x]['type']=='discrete']
    if 'intercept' in discrete:
        discrete.remove('intercept')
    discrete = np.sort(discrete)
    X = glm.design.get_X(kernels=discrete).values
    corr = np.corrcoef(X.T) # remove intercept
    plt.figure(figsize=(10,10))
    p = plt.gca().imshow(corr,cmap='Blues')
    plt.gcf().colorbar(p, ax=plt.gca())
    plt.title('Discrete Regressors')
    plt.xlabel('Regressor')
    plt.ylabel('Regressor')

    # Add ticks to mark each kernel
    start = 0
    end = -1
    ticks =[]
    locs = []
    for x in discrete:
        end += glm.design.kernel_dict[x]['kernel_length_samples'] 
        ticks.append(x)
        locs.append(np.mean([start,end]))
        start += glm.design.kernel_dict[x]['kernel_length_samples'] 
        if add_lines:
            plt.gca().axvline(end+0.5,color='k',alpha=0.05)
            plt.gca().axhline(end+0.5,color='k',alpha=0.05)
    plt.xticks(ticks=locs, labels=ticks,rotation=90)
    plt.yticks(ticks=locs, labels=ticks)
    plt.tight_layout()
    if save_plot:
        plt.savefig('discrete.png')

    # Look at the continuous kernels
    cont = [x for x in glm.run_params['kernels'] if glm.run_params['kernels'][x]['type']=='continuous']
    if 'intercept' in cont:
        cont.remove('intercept')
    cont = np.sort(cont)
    X = glm.design.get_X(kernels=cont).values
    corr = np.corrcoef(X.T) # remove intercept
    plt.figure(figsize=(10,10))
    p = plt.gca().imshow(corr,cmap='Blues')
    plt.gcf().colorbar(p, ax=plt.gca())
    plt.title('Continuous Regressors')
    plt.xlabel('Regressor')
    plt.ylabel('Regressor')
    
    # Add ticks to mark each kernel
    start = 0
    end = -1
    ticks =[]
    locs = []
    for x in cont:
        end += glm.design.kernel_dict[x]['kernel_length_samples'] 
        ticks.append(x)
        locs.append(np.mean([start,end]))
        start += glm.design.kernel_dict[x]['kernel_length_samples'] 
        if add_lines:
            plt.gca().axvline(end+0.5,color='k',alpha=0.05)
            plt.gca().axhline(end+0.5,color='k',alpha=0.05)
    plt.xticks(ticks=locs, labels=ticks,rotation=90)
    plt.yticks(ticks=locs, labels=ticks)
    plt.tight_layout() 
    if save_plot: 
        plt.savefig('continuous.png') 

    # Plot the correlations between the timeseries with no delay for the continuous kernels
    cont_events = np.vstack([glm.design.events[x] for x in cont])
    plt.figure(figsize=(10,10))
    corr = np.corrcoef(cont_events) # remove intercept
    p = plt.gca().imshow(corr,cmap='Blues')
    plt.gcf().colorbar(p, ax=plt.gca())

    # Add faint lines
    for dex,x in enumerate(cont):
        if add_lines:
            plt.gca().axvline(dex+0.5,color='k',alpha=0.05)
            plt.gca().axhline(dex+0.5,color='k',alpha=0.05)

    # Clean up plot and save   
    plt.title('Continuous Timeseries')
    plt.xlabel('Regressors')
    plt.ylabel('Regressors')
    plt.xticks(ticks=range(0,len(cont)), labels=cont,rotation=90)
    plt.yticks(ticks=range(0,len(cont)), labels=cont)
    plt.tight_layout()  
    if save_plot:
        plt.savefig('continuous_events.png') 

def plot_PCA_var_explained(pca, figsize=(10,8)):
    fig,ax=plt.subplots(2,1,figsize=figsize, sharex=True)
    ax[0].plot(
        np.arange(40),
        pca.explained_variance_ratio_,
        'o-k'
    )
    ax[1].plot(
        np.arange(40),
        np.cumsum(pca.explained_variance_ratio_),
        'o-k'
    )

    ax[0].axhline(0, color='gray')
    ax[1].axhline(1, color='gray')
    ax[1].set_xlabel('PC number')
    ax[0].set_ylabel('variance explained')
    ax[1].set_ylabel('cumulative variance explained')
    ax[0].set_title('variance explained by PC')
    ax[1].set_title('cumulative variance explained by PC')
    fig.tight_layout()
    return fig, ax

def pc_component_heatmap(pca, figsize=(18,4)):
    components = pd.DataFrame(pca.components_, columns=pca.component_names)
    sorted_cols = np.array(pca.component_names)[np.argsort(pca.components_[0,:])]
    fig,ax=plt.subplots(figsize=figsize)
    sns.heatmap(
        components[sorted_cols[::-1]].iloc[:10],
        cmap='seismic',
        ax=ax,
        vmin=-1,
        vmax=1
    )
    ax.set_title('Principal Component Vectors')
    ax.set_xticks(np.arange(0.5,len(pca.component_names)+0.5))
    ax.set_xticklabels(sorted_cols[::-1],rotation=45,ha='right')
    ax.set_ylabel('PC number')
    fig.tight_layout()
    return fig, ax

def compare_var_explained(results=None, fig=None, ax=None, figsize=(15,12), outlier_threshold=1.5):
    '''
    make a boxplot comparing variance explained for each version in the database
    inputs:
        results: a dataframe of results (if None, will be retreived from database)
        fig, ax: figure and axis handles. If None, will be created
        figsize: size of figure
        outlier_threshold: Proportion of the IQR past the low and high quartiles to extend the plot whiskers. Points outside this range will be identified as outliers. (from seaborn docs)

    returns:
        figure and axis handles (tuple)
    '''
    if results is None:
        results_dict = gat.retrieve_results()
        results = results_dict['full']
    if fig is None and ax is None:
        fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True, sharex='col')

    cre_line_order = np.sort(results['cre_line'].unique())
    glm_version_order = np.sort(results['glm_version'].unique())

    for row,dataset in enumerate(['train','test']):
        plot1 = sns.boxplot(
            data=results,
            x='glm_version',
            y='Full_avg_cv_var_{}'.format(dataset),
            order = glm_version_order,
            hue='cre_line',
            hue_order=cre_line_order,
            fliersize=0,
            whis=outlier_threshold,
            ax=ax[row,0],
        )

        plot2 = sns.boxplot(
            data=results,
            x='cre_line',
            y='Full_avg_cv_var_{}'.format(dataset),
            order = cre_line_order,
            hue='glm_version',
            hue_order=glm_version_order,
            fliersize=0,
            whis=outlier_threshold,
            ax=ax[row,1],
            palette='brg',
        )
        ax[row, 0].set_ylabel('variance explained')
        ax[row, 0].set_xlabel('GLM version')
        ax[row, 0].set_title('{} data full model performance\ngrouped by version'.format(dataset))
        ax[row, 1].set_title('{} data full model performance\ngrouped by cre line'.format(dataset))

        # calculate interquartile ranges
        grp = results.groupby(['glm_version','cre_line'])['Full_avg_cv_var_{}'.format(dataset)]
        IQR = grp.quantile(0.75) - grp.quantile(0.25)


        lower_bounds = grp.quantile(0.25) - 1.5*IQR
        upper_bounds = grp.quantile(0.75) + 1.5*IQR

        for i in range(2):
            ax[row, i].legend(loc='upper left',bbox_to_anchor=(1.01, 1),borderaxespad=0)
            ax[row, i].set_ylim(lower_bounds.min()-0.05 ,upper_bounds.max()+0.05)
            ax[row, i].axhline(0, color='black', linestyle=':')
            ax[row, i].set_xticklabels(ax[row, i].get_xticklabels(),rotation=30, ha='right')

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('variance explained by GLM version and cre_line (outliers removed from visualization)')

    return fig, ax


def plot_licks(session, ax, y_loc=0, t_span=None):
    if t_span:
        df = session.dataset.licks.query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        df = session.dataset.licks
    ax.plot(
        df['timestamps'],
        y_loc*np.ones_like(df['timestamps']),
        'ok',
        alpha=0.5
    )


def plot_running(session, ax, t_span=None):
    if t_span:
        running_df = session.dataset.running_data_df.reset_index().query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        running_df = session.dataset.running_data_df.reset_index()
    ax.plot(
        running_df['timestamps'],
        running_df['speed'],
        color='blue',
        linewidth=3
    )

def plot_pupil(session, ax, t_span=None):
    '''shares axis with running'''
    vbp.initialize_legend(ax=ax, colors=['blue','black'],linewidth=3)
    if t_span:
        pupil_df = session.dataset.eye_tracking.query(
            'time >= {} and time <= {}'.format(t_span[0], t_span[1]))
    else:
        pupil_df = session.dataset.eye_tracking
    ax.plot(
        pupil_df['time'],
        pupil_df['pupil_area'],
        color='black',
        linewidth=3
    )

    ax.legend(
        ['running','pupil'],
        loc='upper left',
        ncol=10, 
    )



def plot_omissions(session, ax, y_loc=0, t_span=None):
    omissions = session.stimulus_presentations.query('omitted == True')
    ax.plot(
        omissions['start_time'],
        y_loc*np.ones_like(omissions['start_time']),
        marker='*',
        color='red',
        linestyle='none'
    )


def plot_stimuli(session, ax, t_span=None):
    buffer = 0.25
    images = session.dataset.stimulus_presentations['image_name'].unique()
    colors = {image: color for image, color in zip(
        np.sort(images), sns.color_palette("Set2", 8))}

    if t_span:
        query_string = 'start_time >= {0} - {2} and stop_time <= {1} + {2}'.format(
            t_span[0], t_span[1], buffer)
        visual_stimuli = session.dataset.stimulus_presentations.query(
            'omitted == False').query(query_string).copy()
    else:
        visual_stimuli = session.dataset.stimulus_presentations.query(
            'omitted == False').copy()

    visual_stimuli['color'] = visual_stimuli['image_name'].map(
        lambda i: colors[i])
    visual_stimuli['change'] = visual_stimuli['image_name'] != visual_stimuli['image_name'].shift()
    for idx, stimulus in visual_stimuli.iterrows():
        ax.axvspan(
            stimulus['start_time'],
            stimulus['stop_time'],
            color=stimulus['color'],
            alpha=0.5
        )


def build_simulated_FOV(session, F_dataframe, column):

    assert len(session.cell_specimen_table) == len(F_dataframe)

    arr = np.zeros_like(session.dataset.max_projection)
    for ii, cell_specimen_id in enumerate(session.dataset.cell_specimen_ids):

        F_cell = F_dataframe.loc[cell_specimen_id][column]
        arr += session.cell_specimen_table.loc[cell_specimen_id]['image_mask']*F_cell

    return arr


def plot_kernels(kernel_df,ax,t_span=None):
    # kernels_to_exclude_from_plot = []#['intercept','time',]#['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']
    # kernels_to_exclude_from_plot = ['intercept','time',]#['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']
    kernels_to_exclude_from_plot = ['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']

    if t_span:
        t0,t1 = t_span
        data_to_plot = kernel_df.query('timestamps >= @t0 and timestamps <= @t1 and kernel_name not in @kernels_to_exclude_from_plot')
    else:
        data_to_plot = kernel_df.query('kernel_name not in @kernels_to_exclude_from_plot')
    palette = vbp.generate_random_colors(
        len(data_to_plot['kernel_name'].unique()), 
        lightness_range=(0.1,.65), 
        saturation_range=(0.5,1), 
        random_seed=3, 
        order_colors=False
    )
    sns.lineplot(
        data = data_to_plot,
        x='timestamps',
        y='kernel_outputs',
        hue='kernel_name',
        n_boot=0,
        ci=None,
        ax=ax,
        palette = palette,
        alpha=0.75,
        legend=False,
        linewidth=3,
    )
    ax.legend(
        data_to_plot['kernel_name'].unique(),
        loc='upper left',
        ncol=10, 
        mode="expand", 
        framealpha = 0.5,
    )
    # plt.setp(ax.lines,linewidth=4)

def plot_session_summary(glm):
    plt.figure()
    plt.plot(glm.dropout_summary.query('dropout=="Full"')['variance_explained'].sort_values().values)
    plt.axhline(0.00, color='k',alpha=.25)
    plt.axhline(0.01, color='k',alpha=.25)
    plt.gca().axhspan(-.1,0.01, color='k',alpha=0.25)
    plt.ylim(bottom=-.1)
    plt.ylabel('Full Model CV Variance Explained')
    plt.xlabel('Cells')

def plot_dropout_summary(results_summary, cell_specimen_id, ax):
    '''
    makes bar plots of results summary
    inputs:
        glm -- glm object
        cell_specimen_id -- cell to plot
        ax -- a vector of three matplotlib axis handles
    '''
    data_to_plot = (
        results_summary
        .query('cell_specimen_id == @cell_specimen_id')
        .sort_values(by='fraction_change_from_full', ascending=False)
    )

    mixed_dropout_color = 'DimGray'
    special_dropout_colors = {
        'Full':'DarkGreen',
        'beh_model':mixed_dropout_color,
        'all-images':mixed_dropout_color,
        'visual':mixed_dropout_color,
        
    }
    palette = [special_dropout_colors[key] if key in special_dropout_colors else 'black' for key in data_to_plot['dropout']]

    sns.barplot(
        data = data_to_plot,
        x = 'variance_explained',
        y = 'dropout',
        ax=ax[0],
        palette=palette
    )
    sns.barplot(
        data = data_to_plot,
        x = 'absolute_change_from_full',
        y = 'dropout',
        ax=ax[1],
        palette=palette
    )
    sns.barplot(
        data = data_to_plot,
        x = 'fraction_change_from_full',
        y = 'dropout',
        ax=ax[2],
        palette=palette
    )
    ax[0].set_title('variance explained\nfor each model dropout')
    ax[1].set_title('absolute change\nin variance explained')
    ax[2].set_title('fractional change\nin variance explained')
    for col in [1,2]:
        ax[col].set_yticklabels([])
        ax[col].set_ylabel('')

def plot_filters(glm, cell_specimen_id, n_cols=5):
    '''plots all filters for a given cell'''
    kernel_list = list(glm.design.kernel_dict.keys())
    all_weight_names = glm.X.weights.values
    n_rows = int(np.ceil(len(kernel_list)/5))

    fig,ax=plt.subplots(int(n_rows),int(n_cols), figsize=(2.5*n_cols,2.5*n_rows),sharey=True)

    ii = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if ii <= len(kernel_list) - 1:
                kernel_name = kernel_list[ii]
                t = np.linspace(
                    0,
                    glm.design.kernel_dict[kernel_name]['kernel_length_samples']/glm.fit['ophys_frame_rate'],
                    glm.design.kernel_dict[kernel_name]['kernel_length_samples']
                )
                t += glm.design.kernel_dict[kernel_name]['offset_seconds']

                kernel_weight_names = [w for w in all_weight_names if w.startswith(kernel_name)]
                w_kernel = glm.W.loc[dict(weights=kernel_weight_names, cell_specimen_id=cell_specimen_id)]
                ax[row,col].plot(t,w_kernel,marker='.')
                ax[row,col].set_title(kernel_name)
                ax[row,col].axvline(0, color='k',linestyle=':')
                ax[row,col].axhline(0, color='k',linestyle=':')
                
            else:
                ax[row,col].axis('off')
            
            if ii >= len(kernel_list) - n_rows:
                ax[row,col].set_xlabel('time from event (s)')
            if col == 0:
                ax[row,col].set_ylabel('$\Delta$F/F')
            ii += 1

    fig.tight_layout()

    return fig, ax


def get_title(ophys_experiment_id, cell_specimen_id):
    '''
    generate a standardized figure title containing identifying information
    '''
    experiments_table = loading.get_filtered_ophys_experiment_table().reset_index()

    row = experiments_table.query('ophys_experiment_id == @ophys_experiment_id').iloc[0].to_dict()
    title = '{}__specimen_id={}__exp_id={}__{}__{}__depth={}__cell_id={}'.format(
        row['cre_line'],
        row['specimen_id'],
        row['ophys_experiment_id'],
        row['session_type'],
        row['targeted_structure'],
        row['imaging_depth'],
        cell_specimen_id,
    )
    return title

class GLM_Movie(object):

    def __init__(self, glm, cell_specimen_id, start_frame, end_frame, frame_interval=1, fps=10):

        # note that ffmpeg must be installed on your system
        # this is tested on linux (not sure if it works on windows)
        mpl.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
        plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

        plt.style.use('ggplot')

        self.glm = glm
        self.cell_specimen_id = cell_specimen_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_interval = frame_interval

        self.title = get_title(self.glm.oeid, self.cell_specimen_id)

        self.kernel_df = gat.build_kernel_df(self.glm, self.cell_specimen_id)

        self.real_2p_movie = loading.load_motion_corrected_movie(self.glm.oeid)

        self.frames = np.arange(self.start_frame, self.end_frame, self.frame_interval)
        self.fps = fps

        self.results_summary = gat.generate_results_summary(self.glm)
        self.dropout_summary_plotted = False
        self.cell_roi_plotted = False

        self.fig, self.ax = self.set_up_axes()
        self.writer = self.set_up_writer()

    def make_cell_movie_frame(self, ax, glm, F_index, cell_specimen_id, t_before=10, t_after=10):
        # ti = time.time()
        this_cell = glm.df_full.query('cell_specimen_id == @cell_specimen_id')
        cell_index = np.where(glm.W['cell_specimen_id'] == cell_specimen_id)[0][0]

        model_timestamps = glm.fit['dff_trace_arr']['dff_trace_timestamps'].values
        t_now = model_timestamps[F_index]
        t_span = [t_now - t_before, t_now + t_after]
        # print('setup done at {} seconds'.format(time.time() - ti))
        if not self.dropout_summary_plotted:
            plot_dropout_summary(self.results_summary, self.cell_specimen_id, ax['dropout_summary'])
            self.dropout_summary_plotted = True

        for axis_name in ax.keys():
            if axis_name != 'dropout_summary' and axis_name != 'cell_roi':
                ax[axis_name].cla()

        # print('setup done at {} seconds'.format(time.time() - ti))
        F_this_frame = glm.df_full.query('frame_index == @F_index').set_index('cell_specimen_id')
        # dff_actual = dft.loc[glm.W['cell_specimen_id'].values]['dff'].values
        # dff_pred = dft.loc[glm.W['cell_specimen_id'].values]['dff_predicted'].values
        
        # 2P ROI images:
        if not self.cell_roi_plotted:
            ax['cell_roi'].imshow(glm.session.dataset.cell_specimen_table.loc[cell_specimen_id]['image_mask'],cmap='gray')
            self.com = ndimage.measurements.center_of_mass(glm.session.dataset.cell_specimen_table.loc[cell_specimen_id]['image_mask'])
            self.cell_roi_plotted = True

        reconstructed_fov = build_simulated_FOV(glm.session, F_this_frame, 'dff')
        ax['reconstructed_fov'].imshow(reconstructed_fov, cmap='seismic', clim=[-0.5, .5])

        simulated_fov = build_simulated_FOV(glm.session, F_this_frame, 'dff_predicted')
        ax['simulated_fov'].imshow(simulated_fov, cmap='seismic', clim=[-0.5, .5])

        real_fov = self.real_2p_movie[F_index]
        ax['real_fov'].imshow(real_fov, cmap='gray', clim=[0, 15000])

        ax['cell_roi'].set_title('ROI mask for cell {}'.format(cell_specimen_id))
        ax['reconstructed_fov'].set_title('Reconstructed FOV')
        ax['simulated_fov'].set_title('Simulated FOV')
        ax['real_fov'].set_title('Real FOV')

        for axis_name in ['cell_roi','real_fov','reconstructed_fov','simulated_fov']:
            ax[axis_name].set_xticks([])
            ax[axis_name].set_yticks([])
            ax[axis_name].axvline(self.com[1],color='MediumAquamarine',alpha=0.5)
            ax[axis_name].axhline(self.com[0],color='MediumAquamarine',alpha=0.5)

        # time series plots:
        query_string = 'dff_trace_timestamps >= {} and dff_trace_timestamps <= {}'.format(
            t_span[0],
            t_span[1]
        )
        local_df = this_cell.query(query_string)

        ax['cell_response'].plot(
            local_df['dff_trace_timestamps'],
            local_df['dff'],
            alpha=0.5,
            color='darkgreen',
            linewidth=3,
        )

        ax['cell_response'].plot(
            local_df['dff_trace_timestamps'],
            local_df['dff_predicted'],
            alpha=1,
            color='black',
            linewidth=3,
        )

        ax['cell_response'].legend(
            ['Actual $\Delta$F/F','Model Predicted $\Delta$F/F'],
            loc='upper left',
            ncol=2, 
            framealpha = 0.2,
        )

        plot_licks(glm.session, ax['licks'], t_span=t_span)
        plot_running(glm.session, ax['running'], t_span=t_span)
        plot_pupil(glm.session, ax['pupil'], t_span=t_span)
        plot_kernels(self.kernel_df, ax['kernel_contributions'], t_span)

        # some axis formatting: 
        for axis_name in ['licks', 'cell_response', 'running','kernel_contributions']:
            ax[axis_name].axvline(t_now, color='black', linewidth=3, alpha=0.5)
            plot_stimuli(glm.session, ax[axis_name], t_span=t_span)
            if axis_name != 'kernel_contributions':
                ax[axis_name].set_xticklabels([])

        # ax['running'].set_ylim(
        #     self.glm.session.dataset.running_data_df['speed'].min(),
        #     self.glm.session.dataset.running_data_df['speed'].max()
        # )
        # ax['pupil'].set_ylim(
        #     self.glm.session.dataset.eye_tracking['pupil_area'].min(),
        #     self.glm.session.dataset.eye_tracking['pupil_area'].max()
        # )

        # ax['cell_response'].set_ylim(
        #     glm.df_full['dff_predicted'].min(),
        #     glm.df_full['dff_predicted'].max()
        # )

        ax['cell_response'].set_title('Time series plots for cell {}'.format(cell_specimen_id))
        ax['licks'].set_xlim(t_span[0], t_span[1])
        ax['licks'].set_yticks([])

        ax['cell_response'].set_xticklabels('')

        ax['licks'].set_xlabel('time')

        ax['licks'].set_ylabel('licks       ', rotation=0,ha='right', va='center')
        ax['cell_response'].set_ylabel('$\Delta$F/F', rotation=0, ha='right', va='center')
        ax['running'].set_ylabel('Running\nSpeed\n(cm/s)', rotation=0, ha='right', va='center')
        ax['pupil'].set_ylabel('Pupil\nDiameter\n(pix^2)', rotation=0, ha='left', va='center')
        ax['kernel_contributions'].set_ylabel('kernel\ncontributions\nto predicted\nsignal\n($\Delta$F/F)', rotation=0, ha='right', va='center')


    def update(self, frame_number):
        '''
        method to update figure
        animation class will call this

        the print statement is there to help track progress
        '''
        self.make_cell_movie_frame(
            self.ax, self.glm, F_index=frame_number, cell_specimen_id=self.cell_specimen_id)

        self.pbar.update(1)
        gc.collect()

    def set_up_axes(self):
        fig = plt.figure(figsize=(24, 18))
        ax = {
            'cell_roi': vbp.placeAxesOnGrid(fig, xspan=(0, 0.25), yspan=(0, 0.25)),
            'real_fov': vbp.placeAxesOnGrid(fig, xspan=(0.25, 0.5), yspan=(0, 0.25)),
            'reconstructed_fov': vbp.placeAxesOnGrid(fig, xspan=(0.5, 0.75), yspan=(0, 0.25)),
            'simulated_fov': vbp.placeAxesOnGrid(fig, xspan=(0.75, 1), yspan=(0, 0.25)),
            'cell_response': vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.3, 0.45]),
            'licks': vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.45, 0.475]),
            'running': vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.475, 0.575]),
            'kernel_contributions':vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.575, 0.70]),
            'dropout_summary':vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.775, 1], dim=[1,3], wspace=0.01),
        }
        ax['pupil'] = ax['running'].twinx()

        ax['licks'].get_shared_x_axes().join(ax['licks'], ax['cell_response'])
        ax['running'].get_shared_x_axes().join(ax['running'], ax['cell_response'])
        ax['kernel_contributions'].get_shared_x_axes().join(ax['kernel_contributions'], ax['cell_response'])

        variance_explained_string = 'Variance explained (full model) = {:0.1f}%'.format(100*self.glm.results.loc[self.cell_specimen_id]['Full_avg_cv_var_test'])
        fig.suptitle(self.title+'\n'+variance_explained_string)

        return fig, ax

    def set_up_writer(self):

        writer = animation.FFMpegWriter(
            fps=self.fps,
            codec='mpeg4',
            bitrate=-1,
            extra_args=['-pix_fmt', 'yuv420p', '-q:v', '5']
        )
        return writer

    def make_movie(self):
        self.dropout_summary_plotted = False
        self.cell_roi_plotted = False

        a = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=1/self.fps*1000,
            repeat=False,
            blit=False
        )

        base_path = self.glm.run_params['output_dir'].split('/v_')[0]
        save_folder = os.path.join(base_path, 'output_files')
        if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)

        filename = self.title+'_frame_{}_to_{}.mp4'.format(self.start_frame, self.end_frame)

        with tqdm(total=len(self.frames)) as self.pbar:
            a.save(
                os.path.join(save_folder, filename),
                writer=self.writer
            )

def get_containing_dictionary(key,dicts,run_params):
    '''
        Helper function for plot_dropouts()
        returns which dropout contains each kernel
    '''
    label='-'
    
    for d in dicts:
        found=False
        if (d == 'Full') & (key in run_params['dropouts']['Full']['kernels']):
            if found:
                print('WARNING DUPLICATE DROPOUT')
            found=True
            label= d
        elif key in run_params['dropouts'][d]['dropped_kernels']:
            if found:
                print('WARNING DUPLICATE DROPOUT')
            found=True
            label= d
    return label

def make_level(df, drops, this_level_num,this_level_drops,run_params):
    '''
        Helper function for plot_dropouts()
        Determines what dropout each kernel is a part of, as well as keeping track of which dropouts have been used. 
    '''
    df['level-'+str(this_level_num)] = [get_containing_dictionary(key, this_level_drops,run_params) for key in df.index.values]
    for d in this_level_drops:
        drops.remove(d)
    return df,drops

def plot_dropouts(run_params,save_results=False,num_levels=6):
    '''
        Makes a visual and graphic representation of how the kernels are nested inside dropout models
    '''
    if num_levels==4:
        plt.figure(figsize=(16,8))
    elif num_levels==6:
        plt.figure(figsize=(19,8))
    else:
        plt.figure(figsize=(16,8))
    w = 1/num_levels
    
    # Get list of dropouts and kernels
    drops = set([x for x in run_params['dropouts'] if not run_params['dropouts'][x]['is_single'] ])
    kernels = run_params['kernels'].copy()
 
    # Build dataframe
    df = pd.DataFrame(index=kernels.keys())
    
    # Add the individual dropouts
    df['level-1']= df.index.values
    for k in kernels:
        if k in drops:
            drops.remove(k)
    
    # Add each grouping of dropouts
    levels={
            num_levels:['Full'],
            num_levels-1:['visual','behavioral','cognitive'],
            num_levels-2:['licking','task','face_motion_energy','pupil_and_running','all-images','beh_model','expectation'],
            num_levels-3:['licking_bouts','licking_each_lick','pupil_and_omissions','trial_type','change_and_rewards'],
            num_levels-4:['running_and_omissions','hits_and_rewards'],
        }
    for level in np.arange(num_levels,1,-1):
        df,drops = make_level(df,drops, level,  levels[level],  run_params)
        
    # re-organized dataframe
    df=df[['level-'+str(x) for x in range(1,num_levels+1)]]
    df = df.sort_values(by=['level-'+str(x) for x in np.arange(num_levels,0,-1)])
    df['text'] = [run_params['kernels'][k]['text'] for k in df.index.values]
    df['support'] = [(np.round(run_params['kernels'][k]['offset'],2), np.round(run_params['kernels'][k]['length'] +  run_params['kernels'][k]['offset'],2)) for k in df.index.values]

    # Make sure all dropouts were used
    if len(drops) > 0:
        print('Warning, dropouts not used')
        print(drops)

    # Make Color Dictionary
    labels=[]
    colors=[]
    for level in range(1,num_levels+1):
        new_labels = list(df['level-'+str(level)].unique())
        labels = labels + ['level-'+str(level)+'-'+ x for x in new_labels]
        colors = colors + sns.color_palette('hls', len(new_labels)) 
    color_dict = {x:y for (x,y) in  zip(labels,colors)}
    for level in range(1,num_levels+1):
        color_dict['level-'+str(level)+'--'] = (0.8,0.8,0.8)
    
    # Plot Squares
    uniques = set()
    maxn = len(df)
    last = {x:'null' for x in np.arange(1,num_levels+1,1)} 
    for index, k in enumerate(df.index.values):
        for level in range(1,num_levels+1):
            plt.axhspan(maxn-index-1,maxn-index,w*(level-1),w*level,color=color_dict['level-'+str(level)+'-'+df.loc[k]['level-'+str(level)]]) 
            # If this is a new group, add a line and a text label
            if (level > 1)&(not (df.loc[k]['level-'+str(level)] == '-')) & ('level-'+str(level)+'-'+df.loc[k]['level-'+str(level)] not in uniques) :
                uniques.add('level-'+str(level)+'-'+df.loc[k]['level-'+str(level)])
                plt.text(w*(level-1)+0.01,maxn-index-1+.25,df.loc[k]['level-'+str(level)],fontsize=12)
                plt.plot([0,w*level],[maxn-index,maxn-index], 'k-',alpha=1)
                #plt.plot([w*(level-1),w*level],[maxn-index,maxn-index], 'k-',alpha=1)
            elif (level > 1) & (not (df.loc[k]['level-'+str(level)] == last[level])):
                plt.plot([0,w*level],[maxn-index,maxn-index], 'k-',alpha=1)
                #plt.plot([w*(level-1),w*level],[maxn-index,maxn-index], 'k-',alpha=1)
            elif level == 1:
                # For the individual regressors, just label, no lines
                plt.text(0.01,maxn-index-1+.25,df.loc[k]['level-'+str(level)],fontsize=12)
            last[level] = df.loc[k]['level-'+str(level)]

    # Define some lines between levels   
    for level in range(1,num_levels): 
        plt.axvline(w*level,color='k') 
    
    # Make formated ylabels that include support and alignment event   
    max_name = np.max([len(x) for x in df.index.values])+3 
    max_support = np.max([len(str(x)) for x in df['support'].values])+3
    max_text = np.max([len(str(x)) for x in df['text'].values])
    aligned_names = [row[1].name.ljust(max_name)+str(row[1]['support']).ljust(max_support)+row[1]['text'].ljust(max_text) for row in df.iterrows()]

    # clean up axes
    plt.ylim(0,len(kernels))
    plt.xlim(0,1)
    labels = ['Individual Model']+['Minor Component']*(num_levels-3)+['Major Component','Full Model']
    plt.xticks([w*x for x in np.arange(0.5,num_levels+0.5,1)],labels,fontsize=12)
    plt.yticks(np.arange(len(kernels)-0.5,-0.5,-1),aligned_names,ha='left',family='monospace')
    plt.gca().get_yaxis().set_tick_params(pad=400)
    plt.title('Nested Models')
    plt.tight_layout()
    plt.text(-.255,len(kernels)+.35,'Alignment',fontsize=12)
    plt.text(-.385,len(kernels)+.35,'Support',fontsize=12)
    plt.text(-.555,len(kernels)+.35,'Kernel',fontsize=12)
        
    # Save results
    if save_results:
        plt.savefig(run_params['output_dir']+'/nested_models_'+str(num_levels)+'.png')
        df.to_csv(run_params['output_dir']+'/kernels_and_dropouts.csv')
    return df

def all_kernels_evaluation_old(results, run_params):
    '''
        OLD, do not use
        Makes the analysis plots for all kernels in this model version
    '''
    Ws, oeids = get_weights_for_sessions(results, run_params['version'])
    kernels = set(run_params['kernels'].keys())
    kernels.remove('intercept')
    kernels.remove('time')
    for k in kernels:
        try:
            Ws,oeids, weights,cell_data,cell_dropout_data = kernel_evaluation(results,run_params, k,Ws=Ws,oeids=oeids)
        except:
            print(k) 

def kernel_evaluation_old(results,run_params, kernel,save_results=True,Ws=None, oeids=None):
    '''
        OLD, do not use
        Get all the kernels across all cells. 
        plot the matrix of all kernels, sorted by peak time
        plot the mean+std. What time point are different from 0?
        Plot a visualization of the dropouts that contain this kernel. 
    '''

    # Make list of sessions
    #results = gat.retrieve_results(search_dict={'glm_version':'6_L2_optimize_by_session'}, results_type='summary')   

    version = run_params['version'] 
    # Getting Data
    if (Ws is None) or (oeids is None):
        Ws, oeids = get_weights_for_sessions(results, version)
    weights,weight_names,cell_data = get_weights_for_kernel(Ws,oeids, kernel)
    cell_data = pd.merge(cell_data, results[['ophys_experiment_id','cell_specimen_id','cre_line','session_number']], 
                                    how='inner',on=['cell_specimen_id','ophys_experiment_id']).drop_duplicates()
    cell_dropout_data = pd.merge(cell_data, results[['ophys_experiment_id','cell_specimen_id','dropout','adj_fraction_change_from_full']], 
                                    how='inner',on=['cell_specimen_id','ophys_experiment_id'])
    time_vec = np.round(np.array([int(x.split('_')[-1]) for x in weight_names])*(1/31),2) # HARD CODE HACK ALERT

    # Plotting settings
    colors=['C0','C1','C2']
    line_alpha = 0.25
    width=0.25

    # Plotting
    fig,ax=plt.subplots(2,3,figsize=(12,6))
    sst_weights = weights[:,cell_data['cre_line'] == 'Sst-IRES-Cre']
    vip_weights = weights[:,cell_data['cre_line'] == 'Vip-IRES-Cre']
    slc_weights = weights[:,cell_data['cre_line'] == 'Slc17a7-IRES2-Cre']
    ax[0,0].fill_between(time_vec, sst_weights.mean(axis=1)-sst_weights.std(axis=1), sst_weights.mean(axis=1)+sst_weights.std(axis=1),facecolor=colors[0], alpha=0.1)   
    ax[0,0].fill_between(time_vec, vip_weights.mean(axis=1)-vip_weights.std(axis=1), vip_weights.mean(axis=1)+vip_weights.std(axis=1),facecolor=colors[1], alpha=0.1)    
    ax[0,0].fill_between(time_vec, slc_weights.mean(axis=1)-slc_weights.std(axis=1), slc_weights.mean(axis=1)+slc_weights.std(axis=1),facecolor=colors[2], alpha=0.1)    
    ax[0,0].plot(time_vec, sst_weights.mean(axis=1),label='SST',color=colors[0])
    ax[0,0].plot(time_vec, vip_weights.mean(axis=1),label='VIP',color=colors[1])
    ax[0,0].plot(time_vec, slc_weights.mean(axis=1),label='SLC',color=colors[2])
    ax[0,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[0,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[0,0].set_ylabel('Weights (df/f)')
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].legend()
    ax[0,0].set_title('Average kernel')

    sst_weights_norm = sst_weights/np.max(np.abs(sst_weights),axis=0)
    vip_weights_norm = vip_weights/np.max(np.abs(vip_weights),axis=0)
    slc_weights_norm = slc_weights/np.max(np.abs(slc_weights),axis=0)
    ax[1,0].fill_between(time_vec, sst_weights_norm.mean(axis=1)-sst_weights_norm.std(axis=1), sst_weights_norm.mean(axis=1)+sst_weights_norm.std(axis=1),facecolor=colors[0],alpha=0.1) 
    ax[1,0].fill_between(time_vec, vip_weights_norm.mean(axis=1)-vip_weights_norm.std(axis=1), vip_weights_norm.mean(axis=1)+vip_weights_norm.std(axis=1),facecolor=colors[1],alpha=0.1)
    ax[1,0].fill_between(time_vec, slc_weights_norm.mean(axis=1)-slc_weights_norm.std(axis=1), slc_weights_norm.mean(axis=1)+slc_weights_norm.std(axis=1),facecolor=colors[2],alpha=0.1) 
    ax[1,0].plot(time_vec, sst_weights_norm.mean(axis=1),label='SST',color=colors[0])
    ax[1,0].plot(time_vec, vip_weights_norm.mean(axis=1),label='VIP',color=colors[1])
    ax[1,0].plot(time_vec, slc_weights_norm.mean(axis=1),label='SLC',color=colors[2])
    ax[1,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[1,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[1,0].set_ylabel('Weights (df/f)')
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].legend()
    ax[1,0].set_title('Normalized Avg. kernel')
 
    argmax = np.argmax(weights,axis=0)
    argmax[cell_data['cre_line'] == 'Sst-IRES-Cre'] = argmax[cell_data['cre_line'] == 'Sst-IRES-Cre']+1000
    argmax[cell_data['cre_line'] == 'Vip-IRES-Cre'] = argmax[cell_data['cre_line'] == 'Vip-IRES-Cre']+2000
    sort_index = np.argsort(argmax)
    weights_sorted = weights[:,sort_index]
    first_sst = np.where(np.sort(argmax)[::-1] < 1000)[0][0]
    first_vip = np.where(np.sort(argmax)[::-1] < 2000)[0][0]
    cbar = ax[0,1].imshow(weights_sorted.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights)[1]],cmap='bwr')
    ax[0,1].axhline(first_sst,color='k',linewidth='1')
    ax[0,1].axhline(first_vip,color='k',linewidth='1')
    cbar.set_clim(-np.percentile(np.abs(weights),95),np.percentile(np.abs(weights),95))
    fig.colorbar(cbar, ax=ax[0,1])
    ax[0,1].set_ylabel('Cells')
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].set_yticks([first_sst/2, first_sst+(first_vip-first_sst)/2, first_vip+(len(argmax)-first_vip)/2])
    ax[0,1].set_yticklabels(['Vip','Sst','Slc'])
    ax[0,1].set_title(kernel)
   
    # Dropout Scores 
    ax[1,2].tick_params(top='off',bottom='off', left='off',right='off')
    ax[1,2].set_xticks([])
    ax[1,2].set_yticks([])
    for spine in ax[1,2].spines.values():
        spine.set_visible(False)

    # Make list of dropouts 
    drop_list = [d for d in run_params['dropouts'].keys() if ((run_params['dropouts'][d]['is_single']) & (kernel in run_params['dropouts'][d]['kernels'])) or ((not run_params['dropouts'][d]['is_single']) & (kernel in run_params['dropouts'][d]['dropped_kernels']))]
    medianprops = dict(color='k')

    # For each dropout, plot score
    for index, dropout in enumerate(drop_list):
        drop_sst = cell_dropout_data.query('(dropout == @dropout)&(cre_line=="Sst-IRES-Cre")')['adj_fraction_change_from_full'].values
        drop_vip = cell_dropout_data.query('(dropout == @dropout)&(cre_line=="Vip-IRES-Cre")')['adj_fraction_change_from_full'].values
        drop_slc = cell_dropout_data.query('(dropout == @dropout)&(cre_line=="Slc17a7-IRES2-Cre")')['adj_fraction_change_from_full'].values
        drops = ax[0,2].boxplot([drop_sst,drop_vip,drop_slc],positions=[index-width,index,index+width],labels=['SST','VIP','SLC'],showfliers=False,patch_artist=True,medianprops=medianprops,widths=.2)
        for patch, color in zip(drops['boxes'],colors):
            patch.set_facecolor(color)

    ax[0,2].set_ylabel('Adj. Fraction from Full')
    ax[0,2].set_xticks(np.arange(0,len(drop_list)))
    ax[0,2].set_xticklabels(drop_list,rotation=60)
    ax[0,2].axhline(0,color='k',linestyle='--',alpha=line_alpha)
    ax[0,2].set_ylim(-1.05,.05)
    ax[0,2].set_title('Dropout Scores')

    # Plot normalized things
    weights_sorted_norm = weights_sorted/np.max(np.abs(weights_sorted),axis=0)
    cbar = ax[1,1].imshow(weights_sorted_norm.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights)[1]],cmap='bwr')
    ax[1,1].axhline(first_sst,color='k',linewidth='1')
    ax[1,1].axhline(first_vip,color='k',linewidth='1')
    cbar.set_clim(-np.max(np.abs(weights)),np.max(np.abs(weights)))
    fig.colorbar(cbar, ax=ax[1,1])
    ax[1,1].set_ylabel('Cells')
    ax[1,1].set_xlabel('Time (s)')
    ax[1,1].set_yticks([first_sst/2, first_sst+(first_vip-first_sst)/2, first_vip+(len(argmax)-first_vip)/2])
    ax[1,1].set_yticklabels(['Vip','Sst','Slc'])
    ax[1,1].set_title('Normalized '+kernel)

    plt.tight_layout()
    if save_results:
        plt.savefig(run_params['output_dir']+'/'+kernel+'_analysis.png')
    return Ws, oeids, weights,cell_data,cell_dropout_data
 


def get_weights_for_kernel(Ws,oeids, kernel):
    '''
        OLD, do not use
    '''
    include = [i for i in range(0,len(Ws)) if len([w for w in Ws[i].weights.values if w.startswith(kernel)])>0]
    weight_names = [w for w in Ws[0].weights.values if w.startswith(kernel)]
    kernel_weights = [Ws[i].loc[dict(weights=weight_names)].values for i in include]
    cells = [Ws[i].cell_specimen_id.values for i in include]
    oeid_per_cell = [[x]*len(y) for (x,y) in zip(oeids[include], cells)]
    cell_data = pd.DataFrame()
    cell_data['cell_specimen_id'] = np.hstack(cells)
    cell_data['ophys_experiment_id'] = np.hstack(oeid_per_cell)
    return np.hstack(kernel_weights),weight_names, cell_data

def get_weights_for_sessions(results,version):
    '''
        OLD, do not use
    '''
    oeids = results.query('(equipment_name in ["CAM2P.3","CAM2P.4","CAM2P.5"])&(session_number in [1,3,4,6])').ophys_experiment_id.unique()[0:5]
    Ws = []
    for index, oeid in enumerate(oeids):
        W = gat.get_weights_matrix_from_mongo(int(oeid), version) 
        Ws.append(W)
    return Ws,oeids

def process_session_to_df(oeid, run_params):
    '''
        For the ophys_experiment_id, loads the weight matrix, and builds a dataframe
        organized by cell_id and kernel 
    '''
    W = gat.get_weights_matrix_from_mongo(int(oeid), run_params['version'])
    session_df  = pd.DataFrame()
    session_df['cell_specimen_id'] = W.cell_specimen_id.values
    session_df['ophys_experiment_id'] = [int(oeid)]*len(W.cell_specimen_id.values)  
    for k in run_params['kernels']:
        weight_names = [w for w in W.weights.values if w.startswith(k)]
        if len(weight_names) > 0:
            session_df[k] = W.loc[dict(weights=weight_names)].values.T.tolist()
    return session_df

def build_weights_df(run_params,results_pivoted, cache_results=False,load_cache=False):
    '''
        Builds a dataframe of (cell_specimen_id, ophys_experiment_id) with the weight parameters for each kernel
        Some columns may have NaN if that cell did not have a kernel, for example if a missing datastream   
 
        INPUTS:
        run_params, parameter json for the version to analyze
        results_pivoted = gat.build_pivoted_results_summary('adj_fraction_change_from_full',results_summary=results)
        cache_results, if True, save dataframe as csv file
        load_cache, if True, load cached results, if it exists
    
        RETURNS:
        a dataframe
    '''
    
    if load_cache & os.path.exists(run_params['output_dir']+'/weights_df.csv'):
        # Need to convert things to np.array
        return pd.read_csv(run_params['output_dir']+'/weights_df.csv')
   
    # Make dataframe for cells and experiments 
    #session_info=results_pivoted[['cell_specimen_id','ophys_experiment_id','cre_line','session_number','equipment_name','variance_explained_full']]#.drop_duplicates() 
    #results_pivoted = results_pivoted    
    oeids = results_pivoted['ophys_experiment_id'].unique() 

    # For each experiment, get the weight matrix from mongo (slow)
    # Then pull the weights from each kernel into a dataframe
    sessions = []
    for index, oeid in enumerate(tqdm(oeids)):
        session_df = process_session_to_df(oeid, run_params)
        sessions.append(session_df)

    # Merge all the session_dfs, and add more session level info
    weights_df = pd.concat(sessions,sort=False)
    weights_df = pd.merge(weights_df,results_pivoted, on = ['cell_specimen_id','ophys_experiment_id'],suffixes=('_weights',''))
    
    # Cache Results
    if cache_results:
        weights_df.to_csv(run_params['output_dir']+'/weights_df.csv') 

    # Return weights_df
    return weights_df 

def kernel_evaluation(weights_df, run_params, kernel, save_results=True):
    '''
        Get all the kernels across all cells. 
        plot the matrix of all kernels, sorted by peak time
        plot the mean+std. What time point are different from 0?
        Plot a visualization of the dropouts that contain this kernel. 
        results_pivoted = gat.build_pivoted_results_summary('adj_fraction_change_from_full',results_summary=results)
    '''
   
    # Filter out Mesoscope and make time basis 
    version = run_params['version'] 
    weights = weights_df.query('equipment_name in ["CAM2P.3","CAM2P.4","CAM2P.5"]') 
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/31)
    time_vec = np.round(time_vec,2)

    # Plotting settings
    colors=['C0','C1','C2']
    line_alpha = 0.25
    width=0.25
    threshold = 0.01

    # Plot Average Trajectories
    fig,ax=plt.subplots(3,3,figsize=(12,9))
    sst_weights = weights.query('cre_line == "Sst-IRES-Cre"')[kernel+'_weights']
    vip_weights = weights.query('cre_line == "Vip-IRES-Cre"')[kernel+'_weights']
    slc_weights = weights.query('cre_line == "Slc17a7-IRES2-Cre"')[kernel+'_weights']
    sst = np.vstack([x for x in sst_weights[~sst_weights.isnull()].values])
    vip = np.vstack([x for x in vip_weights[~vip_weights.isnull()].values])
    slc = np.vstack([x for x in slc_weights[~slc_weights.isnull()].values])
    ax[0,0].fill_between(time_vec, sst.mean(axis=0)-sst.std(axis=0), sst.mean(axis=0)+sst.std(axis=0),facecolor=colors[0], alpha=0.1)   
    ax[0,0].fill_between(time_vec, vip.mean(axis=0)-vip.std(axis=0), vip.mean(axis=0)+vip.std(axis=0),facecolor=colors[1], alpha=0.1)    
    ax[0,0].fill_between(time_vec, slc.mean(axis=0)-slc.std(axis=0), slc.mean(axis=0)+slc.std(axis=0),facecolor=colors[2], alpha=0.1)    
    ax[0,0].plot(time_vec, sst.mean(axis=0),label='SST',color=colors[0])
    ax[0,0].plot(time_vec, vip.mean(axis=0),label='VIP',color=colors[1])
    ax[0,0].plot(time_vec, slc.mean(axis=0),label='SLC',color=colors[2])
    ax[0,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[0,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[0,0].set_ylabel('Weights (df/f)')
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].legend()
    ax[0,0].set_title('Average kernel')

    sst = sst.T
    vip = vip.T
    slc = slc.T
    sst_weights_norm = sst/np.max(np.abs(sst),axis=0)
    vip_weights_norm = vip/np.max(np.abs(vip),axis=0)
    slc_weights_norm = slc/np.max(np.abs(slc),axis=0)
    ax[1,0].fill_between(time_vec, sst_weights_norm.mean(axis=1)-sst_weights_norm.std(axis=1), sst_weights_norm.mean(axis=1)+sst_weights_norm.std(axis=1),facecolor=colors[0],alpha=0.1) 
    ax[1,0].fill_between(time_vec, vip_weights_norm.mean(axis=1)-vip_weights_norm.std(axis=1), vip_weights_norm.mean(axis=1)+vip_weights_norm.std(axis=1),facecolor=colors[1],alpha=0.1)
    ax[1,0].fill_between(time_vec, slc_weights_norm.mean(axis=1)-slc_weights_norm.std(axis=1), slc_weights_norm.mean(axis=1)+slc_weights_norm.std(axis=1),facecolor=colors[2],alpha=0.1) 
    ax[1,0].plot(time_vec, sst_weights_norm.mean(axis=1),label='SST',color=colors[0])
    ax[1,0].plot(time_vec, vip_weights_norm.mean(axis=1),label='VIP',color=colors[1])
    ax[1,0].plot(time_vec, slc_weights_norm.mean(axis=1),label='SLC',color=colors[2])
    ax[1,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[1,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[1,0].set_ylabel('Weights (df/f)')
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].legend()
    ax[1,0].set_title('Normalized Avg. kernel')

    sst_weights_filtered = weights.query('cre_line == "Sst-IRES-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    vip_weights_filtered = weights.query('cre_line == "Vip-IRES-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    slc_weights_filtered = weights.query('cre_line == "Slc17a7-IRES2-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    sst_f = np.vstack([x for x in sst_weights_filtered[~sst_weights_filtered.isnull()].values])
    vip_f = np.vstack([x for x in vip_weights_filtered[~vip_weights_filtered.isnull()].values])
    slc_f = np.vstack([x for x in slc_weights_filtered[~slc_weights_filtered.isnull()].values])
    ax[2,0].fill_between(time_vec, sst_f.mean(axis=0)-sst_f.std(axis=0), sst_f.mean(axis=0)+sst_f.std(axis=0),facecolor=colors[0], alpha=0.1)   
    ax[2,0].fill_between(time_vec, vip_f.mean(axis=0)-vip_f.std(axis=0), vip_f.mean(axis=0)+vip_f.std(axis=0),facecolor=colors[1], alpha=0.1)    
    ax[2,0].fill_between(time_vec, slc_f.mean(axis=0)-slc_f.std(axis=0), slc_f.mean(axis=0)+slc_f.std(axis=0),facecolor=colors[2], alpha=0.1)    
    ax[2,0].plot(time_vec, sst_f.mean(axis=0),label='SST',color=colors[0])
    ax[2,0].plot(time_vec, vip_f.mean(axis=0),label='VIP',color=colors[1])
    ax[2,0].plot(time_vec, slc_f.mean(axis=0),label='SLC',color=colors[2])
    ax[2,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[2,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[2,0].set_ylabel('Weights (df/f)')
    ax[2,0].set_xlabel('Time (s)')
    ax[2,0].legend()
    ax[2,0].set_title('Average kernel - Filtered cells')
    sst_f = sst_f.T
    vip_f = vip_f.T
    slc_f = slc_f.T

    # Plot Heat maps
    sst_sorted = sst[:,np.argsort(np.argmax(sst,axis=0))]
    vip_sorted = vip[:,np.argsort(np.argmax(vip,axis=0))]
    slc_sorted = slc[:,np.argsort(np.argmax(slc,axis=0))]

    weights_sorted = np.hstack([slc_sorted,sst_sorted, vip_sorted])
    cbar = ax[0,1].imshow(weights_sorted.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted)[1]],cmap='bwr')
    cbar.set_clim(-np.percentile(np.abs(weights_sorted),95),np.percentile(np.abs(weights_sorted),95))
    fig.colorbar(cbar, ax=ax[0,1])
    ax[0,1].set_ylabel('Cells')
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].axhline(np.shape(vip)[1],color='k',linewidth='1')
    ax[0,1].axhline(np.shape(vip)[1] + np.shape(sst)[1],color='k',linewidth='1')
    ax[0,1].set_yticks([np.shape(vip)[1]/2,np.shape(vip)[1]+np.shape(sst)[1]/2, np.shape(vip)[1]+np.shape(sst)[1]+np.shape(slc)[1]/2])
    ax[0,1].set_yticklabels(['Vip','Sst','Slc'])
    ax[0,1].set_title(kernel)

    # Plot normalized things
    weights_sorted_norm = weights_sorted/np.max(np.abs(weights_sorted),axis=0)
    cbar = ax[1,1].imshow(weights_sorted_norm.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted_norm)[1]],cmap='bwr')
    cbar.set_clim(-np.max(np.abs(weights_sorted_norm)),np.max(np.abs(weights_sorted_norm)))
    fig.colorbar(cbar, ax=ax[1,1])
    ax[1,1].set_ylabel('Cells')
    ax[1,1].set_xlabel('Time (s)')
    ax[1,1].axhline(np.shape(vip)[1],color='k',linewidth='1')
    ax[1,1].axhline(np.shape(vip)[1] + np.shape(sst)[1],color='k',linewidth='1')
    ax[1,1].set_yticks([np.shape(vip)[1]/2,np.shape(vip)[1]+np.shape(sst)[1]/2, np.shape(vip)[1]+np.shape(sst)[1]+np.shape(slc)[1]/2])
    ax[1,1].set_yticklabels(['Vip','Sst','Slc'])
    ax[1,1].set_title('Normalized '+kernel)

    # Plot Heatmap of filtered cells
    sst_sorted_f = sst_f[:,np.argsort(np.argmax(sst_f,axis=0))]
    vip_sorted_f = vip_f[:,np.argsort(np.argmax(vip_f,axis=0))]
    slc_sorted_f = slc_f[:,np.argsort(np.argmax(slc_f,axis=0))]
    weights_sorted_f = np.hstack([slc_sorted_f,sst_sorted_f, vip_sorted_f])
    cbar = ax[2,1].imshow(weights_sorted_f.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted_f)[1]],cmap='bwr')
    cbar.set_clim(-np.percentile(np.abs(weights_sorted_f),95),np.percentile(np.abs(weights_sorted_f),95))
    fig.colorbar(cbar, ax=ax[2,1])
    ax[2,1].set_ylabel('Cells')
    ax[2,1].set_xlabel('Time (s)')
    ax[2,1].axhline(np.shape(vip_f)[1],color='k',linewidth='1')
    ax[2,1].axhline(np.shape(vip_f)[1] + np.shape(sst_f)[1],color='k',linewidth='1')
    ax[2,1].set_yticks([np.shape(vip_f)[1]/2,np.shape(vip_f)[1]+np.shape(sst_f)[1]/2, np.shape(vip_f)[1]+np.shape(sst_f)[1]+np.shape(slc_f)[1]/2])
    ax[2,1].set_yticklabels(['Vip','Sst','Slc'])
    ax[2,1].set_title('Filtered - '+kernel)
 
    # Dropout Scores 
    ax[1,2].tick_params(top='off',bottom='off', left='off',right='off')
    ax[1,2].set_xticks([])
    ax[1,2].set_yticks([])
    for spine in ax[1,2].spines.values():
        spine.set_visible(False)

    # Make list of dropouts 
    drop_list = [d for d in run_params['dropouts'].keys() if ((run_params['dropouts'][d]['is_single']) & (kernel in run_params['dropouts'][d]['kernels'])) or ((not run_params['dropouts'][d]['is_single']) & (kernel in run_params['dropouts'][d]['dropped_kernels']))]
    medianprops = dict(color='k')
    
    # For each dropout, plot score
    for index, dropout in enumerate(drop_list):
        drop_sst = weights.query('cre_line=="Sst-IRES-Cre"')[dropout].values
        drop_vip = weights.query('cre_line=="Vip-IRES-Cre"')[dropout].values
        drop_slc = weights.query('cre_line=="Slc17a7-IRES2-Cre"')[dropout].values
        drops = ax[0,2].boxplot([drop_sst,drop_vip,drop_slc],positions=[index-width,index,index+width],labels=['SST','VIP','SLC'],showfliers=False,patch_artist=True,medianprops=medianprops,widths=.2)
        for patch, color in zip(drops['boxes'],colors):
            patch.set_facecolor(color)

    ax[0,2].set_ylabel('Adj. Fraction from Full')
    ax[0,2].set_xticks(np.arange(0,len(drop_list)))
    ax[0,2].set_xticklabels(drop_list,rotation=60)
    ax[0,2].axhline(0,color='k',linestyle='--',alpha=line_alpha)
    ax[0,2].set_ylim(-1.05,.05)
    ax[0,2].set_title('Dropout Scores')


    #sst_weights_filtered = weights.query('cre_line == "Sst-IRES-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    # For each dropout, plot score
    for index, dropout in enumerate(drop_list):
        drop_sst = weights.query('cre_line=="Sst-IRES-Cre" & variance_explained_full > @threshold')[dropout].values
        drop_vip = weights.query('cre_line=="Vip-IRES-Cre" & variance_explained_full > @threshold')[dropout].values
        drop_slc = weights.query('cre_line=="Slc17a7-IRES2-Cre" & variance_explained_full > @threshold')[dropout].values
        drops = ax[2,2].boxplot([drop_sst,drop_vip,drop_slc],positions=[index-width,index,index+width],labels=['SST','VIP','SLC'],showfliers=False,patch_artist=True,medianprops=medianprops,widths=.2)
        for patch, color in zip(drops['boxes'],colors):
            patch.set_facecolor(color)

    ax[2,2].set_ylabel('Adj. Fraction from Full')
    ax[2,2].set_xticks(np.arange(0,len(drop_list)))
    ax[2,2].set_xticklabels(drop_list,rotation=60)
    ax[2,2].axhline(0,color='k',linestyle='--',alpha=line_alpha)
    ax[2,2].set_ylim(-1.05,.05)
    ax[2,2].set_title('Dropout Scores-Filtered Cells')



    plt.tight_layout()
    if save_results:
        plt.savefig(run_params['output_dir']+'/'+kernel+'_analysis.png')

 
