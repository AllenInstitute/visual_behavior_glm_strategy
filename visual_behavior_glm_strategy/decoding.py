import numpy as np
import pandas as pd
from visual_behavior.data_access import reformat 
import visual_behavior_glm_strategy.GLM_fit_tools as gft
import visual_behavior_glm_strategy.GLM_visualization_tools as gvt
import visual_behavior_glm_strategy.build_dataframes as bd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm

def annotate(session):
    reformat.add_licks_each_flash(session.stimulus_presentations, session.licks) 
    reformat.add_rewards_each_flash(session.stimulus_presentations, session.rewards)
    session.stimulus_presentations['licked'] = [True if len(licks) > 0 else False \
        for licks in session.stimulus_presentations.licks.values]
    session.stimulus_presentations['hit'] =  \
        session.stimulus_presentations['licked'] & \
        session.stimulus_presentations['is_change']
    session.stimulus_presentations['miss'] = \
        ~session.stimulus_presentations['licked'] & \
        session.stimulus_presentations['is_change']
    session.stimulus_presentations.at[\
        ~session.stimulus_presentations['is_change'],'hit'] = np.nan
    session.stimulus_presentations.at[\
        ~session.stimulus_presentations['is_change'],'miss'] = np.nan
    session.stimulus_presentations['FA'] = \
        session.stimulus_presentations['licked'] & \
        ~session.stimulus_presentations['is_change'] &\
        ~session.stimulus_presentations['omitted'] & \
        ~session.stimulus_presentations['licked'].shift(1,fill_value=False) &\
        ~session.stimulus_presentations['licked'].shift(2,fill_value=False) &\
        ~session.stimulus_presentations['omitted'].shift(1,fill_value=False) &\
        ~session.stimulus_presentations['omitted'].shift(2,fill_value=False) &\
        ~session.stimulus_presentations['is_change'].shift(1,fill_value=False) &\
        ~session.stimulus_presentations['is_change'].shift(2,fill_value=False) 
    session.stimulus_presentations['pre_FA'] = \
        session.stimulus_presentations['FA'].shift(-1,fill_value=False)
    return session


def decode_experiment(oeid, version, data='events',window=[0,.4],FA=False):

    # Load SDK object
    print('Loading data')
    run_params = {'include_invalid_rois':False}
    session = gft.load_data(oeid, run_params)  
    session = annotate(session)
 
    # get list of cell dataframes
    print('Generating cell dataframes')
    cells = get_cells(session,data=data,window=window,FA=FA)

    # Perform decoding iterating over the number of cells
    print('Decoding')
    results_df = iterate_n_cells(cells,FA=FA)

    # Save results
    print('Save results')
    filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet'+\
        '/behavior/decoding/experiments_fit_{}/'.format(version)
    if FA: 
        filename += str(oeid)+'_FA.pkl'
    else:
        filename += str(oeid)+'.pkl'
    print('Saving to: '+filename)
    results_df.to_pickle(filename)
    print('Finished')


def load_experiment_results(oeid,version=None,FA=False):
    if version is not None:
        filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet'+\
            '/behavior/decoding/experiments_fit_{}/'.format(version)
    else:
        filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet'+\
            '/behavior/decoding/experiments/'
    if FA:
        filename += str(oeid)+'_FA.pkl'   
    else: 
        filename += str(oeid)+'.pkl'
    return pd.read_pickle(filename)


def load_all(experiment_table,summary_df,version=None,mesoscope_only=False,FA=False):
    
    # Iterate through experiments and load decoding results
    dfs = []
    failed = 0
    if mesoscope_only:
        summary_df = summary_df.query('equipment_name == "MESO.1"').copy()
    oeids = np.concatenate(summary_df['ophys_experiment_id'].values)

    for oeid in oeids:
        try:
            df = load_experiment_results(oeid,version,FA)
        except:
            failed +=1
        else:
            df['ophys_experiment_id'] = oeid
            dfs.append(df)
    print('Failed to load: {}'.format(failed))
    
    # Merge and add experiment meta data
    print('Concatenating')
    df = pd.concat(dfs,sort=True)
    print('merging experiment table')
    df = pd.merge(df,experiment_table.reset_index()[['ophys_experiment_id',\
        'ophys_session_id','behavior_session_id','targeted_structure',\
        'imaging_depth','equipment_name']],on='ophys_experiment_id')
    print('merging summary_df')
    df = pd.merge(df, summary_df[['ophys_session_id','visual_strategy_session',\
        'experience_level','cre_line']],on='ophys_session_id')

    return df


def get_cells(session, data='events',window=[0,.4],FA=False):
    '''
        Iterate over all cells in this experiment and make a list
        of cell dataframes
    '''   
 
    # Iterate over all cells in experiment
    cells = []
    cell_specimen_ids = session.cell_specimen_table.index.values
    for cell in tqdm(cell_specimen_ids):
        # Generate the dataframe for this cell
        cell_df = get_cell_table(session, cell,data=data,window=window,FA=FA)
        cells.append(cell_df)
    
    # return list of cell dataframes
    return cells    
         

def get_cell_table(session, cell_specimen_id, data='events',
    window=[0,.4],balance=True,FA=False):
    '''
        Generates a dataframe for one cell where rows are image presentations
        and the response at each timepoint is a column
    '''
    # Get cell activity interpolated onto constant time points
    df = bd.get_cell_df(session, cell_specimen_id, data=data)
    full_df = bd.get_cell_etr(df, session, time = window)
    
    # Pivot the table such that all responses at the same time point are a column
    cell = pd.pivot_table(full_df, values='response',
        index='stimulus_presentations_id', columns=['time'])
    
    # Annotate stimulus information
    cell = pd.merge(cell, session.stimulus_presentations, 
        left_index=True, right_index=True)

    # Balance classes by using just changes and the image before
    cell['pre_change'] = cell['is_change'].shift(-1,fill_value=False)

    if FA:
        cell = cell.query('FA or pre_FA')
    elif balance:
        cell = cell.query('is_change or pre_change')

    return cell


def get_matrix(cell):
    '''
        Grabs the responses of the cell at each time point across each 
        image presentation and returns a numpy matrix 
    '''
    cols = np.sort([x for x in cell.columns.values if not isinstance(x,str)])
    x = cell[cols].to_numpy()
    return x


def plot_by_strategy_performance(visual, timing,savefig,cell_type):
    # Plot decoder performance
    plt.figure()
    plt.clf()

    # visual decoder performance
    v = visual.groupby('n_cells')
    summary = v.mean()
    summary['size'] = v.size()
    summary['sem_score'] = v['test_score'].sem()
    plt.errorbar(summary.index.values, summary['test_score'],
        yerr=summary['sem_score'],color='darkorange')
    plt.plot(summary.index.values, summary.test_score,'o-',color='darkorange',
        label='visual sessions')

    #timing decoder performance
    t = timing.groupby('n_cells')
    summary = t.mean()
    summary['size'] = t.size()
    summary['sem_score'] = t['test_score'].sem()
    plt.errorbar(summary.index.values, summary['test_score'],
        yerr=summary['sem_score'],color='blue')
    plt.plot(summary.index.values, summary.test_score,'bo-',
        label='timing sessions')

    # Clean up decoder performance plot
    plt.xlim(0,80)
    plt.ylim(0.5,1)
    plt.ylabel('decoder performance',fontsize=16)
    plt.xlabel('number of cells',fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # save decoder performance figure
    if savefig:
        filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet'+\
            '/behavior/decoding/figures/' 
        filename += cell_type+'_decoder_performance.png'
        plt.savefig(filename)


def plot_by_strategy_correlation(visual, timing, savefig, cell_type):
    # Plot behavior correlation figure
    plt.figure()
    plt.clf()   
 
    # visual correlation
    v = visual.groupby('n_cells')
    summary = v.mean()
    summary['size'] = v.size()
    summary['sem_behavior_correlation'] = v['behavior_correlation'].sem()
    plt.errorbar(summary.index.values, summary['behavior_correlation'],
        yerr=summary['sem_behavior_correlation'],color='darkorange')
    plt.plot(summary.index.values, summary.behavior_correlation,'o-',
        color='darkorange', label='visual sessions')

    # timing correlation
    t = timing.groupby('n_cells')
    summary = t.mean()
    summary['size'] = t.size()
    summary['sem_behavior_correlation'] = t['behavior_correlation'].sem()
    plt.errorbar(summary.index.values, summary['behavior_correlation'],
        yerr=summary['sem_behavior_correlation'], color='blue')
    plt.plot(summary.index.values, summary.behavior_correlation,'bo-',
        label='timing sessions')

    # clean up correlation figure
    plt.xlim(0,80)
    plt.ylim(0,0.5)
    plt.ylabel('decoder correlation with behavior',fontsize=16)
    plt.xlabel('number of cells',fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # save behavior correlation figure
    if savefig:
        filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet'+\
            '/behavior/decoding/figures/' 
        filename += cell_type+'_decoder_correlation.png'
        plt.savefig(filename)


def plot_by_strategy_scatter(visual, timing, metric, savefig, cell_type,
    version=None,ax=None,meso=False):

    # Plot behavior correlation figure
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
    colors = gvt.project_colors()
    color = colors[cell_type] 
    
    # get_stats
    df = stats_by_strategy(cell_type,visual,timing,metric)
 
    # visual correlation
    v = visual.groupby('n_cells')
    vsummary = v.mean()
    vsummary['size'] = v.size()
    vsummary['sem_'+metric] = v[metric].sem()

    t = timing.groupby('n_cells')
    tsummary = t.mean()
    tsummary['size'] = t.size()
    tsummary['sem_'+metric] = t[metric].sem()

    n = np.min([len(vsummary), len(tsummary)])

    ax.errorbar(tsummary.iloc[0:n][metric],
        vsummary.iloc[0:n][metric],
        xerr=tsummary.iloc[0:n]['sem_'+metric],
        yerr=vsummary.iloc[0:n]['sem_'+metric],
        color=color,fmt='o',markersize=1,alpha=.5)


    for index in range(0,n):
        ax.scatter(tsummary.iloc[index][metric],
            vsummary.iloc[index][metric],
            s=tsummary.index.values[index]*1.5,
            edgecolors=color,
            facecolors=color,
            label=tsummary.index.values[index],alpha=1,zorder=10)
        
    df = df.set_index('n_cells')
    for n in df.index.values:
        if df.loc[n]['p'] < 0.05:
            ax.plot(tsummary.loc[n][metric]-.005,
                    vsummary.loc[n][metric]+.005,
                    'k*')
    if metric == 'behavior_correlation':
        ax.plot([0, .2],[0, .2], 'k--',alpha=.25)
        ax.set_ylabel('visual sessions',
            fontsize=16)
        ax.set_xlabel('timing sessions',
            fontsize=16)
        if meso:
            ax.set_xlim(0,.2)
            ax.set_ylim(0,.2)
            ticks = np.arange(0,.22,.04)
        else:
            ax.set_xlim(0,.165)
            ax.set_ylim(0,.165)
            ticks = np.arange(0,.18,.02)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title('change decoder correlation with behavior',fontsize=16)
    elif metric == 'test_score':
        ax.plot([0.5, 1],[0.5, 1], 'k--',alpha=.25)
        ax.set_ylabel('visual sessions',
            fontsize=16)
        ax.set_xlabel('timing sessions',
            fontsize=16)
        ax.set_xlim(0.5,.75)
        ax.set_ylim(0.5,.75)     
        ax.set_title('change decoder performance',fontsize=16) 
    elif metric == 'test_score_hit_vs_miss':
        ax.plot([0.5, 1],[0.5, 1], 'k--',alpha=.25)
        ax.set_ylabel('visual sessions',
            fontsize=16)
        ax.set_xlabel('timing sessions',
            fontsize=16)
        if meso:
            ax.set_xlim(0.5,.8)
            ax.set_ylim(0.5,.8)             
        else:
            ax.set_xlim(0.5,.75)
            ax.set_ylim(0.5,.75)      
        ax.set_title('hit decoder performance',fontsize=16)
    elif metric == 'test_score_FA':
        ax.plot([0.5, 1],[0.5, 1], 'k--',alpha=.25)
        ax.set_ylabel('visual sessions',
            fontsize=16)
        ax.set_xlabel('timing sessions',
            fontsize=16)
        if meso:
            ax.set_xlim(0.5,.8)
            ax.set_ylim(0.5,.8)            
            ax.set_xlim(0.5,.55)
            ax.set_ylim(0.5,.55)            
        else:
            ax.set_xlim(0.5,.75)
            ax.set_ylim(0.5,.75)      
        ax.set_title('FA decoder performance',fontsize=16) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)



    ax.set_aspect('equal')
    ax.legend(loc='upper left',bbox_to_anchor=(1.04,1),title='# of cells')
    plt.tight_layout()

    if savefig:
        if version is None:
            filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet'+\
                '/behavior/decoding/figures/' 
        else:
            filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet'+\
                '/behavior/decoding/figures_fit_{}/'.format(version)
        if meso:
            filename += 'scatter_by_cre_meso_'+metric+'.svg'  
        else:
            filename += 'scatter_by_cre_'+metric+'.svg'
        print(filename)
        plt.savefig(filename)
    return df

def plot_by_strategy_hit_vs_miss(visual, timing, savefig, cell_type):

    # Plot hit vs miss decoder performance
    plt.figure()
    plt.clf()

    # visual hit vs miss decoder performance
    v = visual.groupby('n_cells')
    summary = v.mean()
    summary['size'] = v.size()
    summary['sem_score'] = v['test_score_hit_vs_miss'].sem()
    plt.errorbar(summary.index.values, summary['test_score_hit_vs_miss'],
        yerr=summary['sem_score'],color='darkorange')
    plt.plot(summary.index.values, summary.test_score_hit_vs_miss,'o-',
        color='darkorange',label='visual sessions')

    #timing hit vs miss decoder performance
    t = timing.groupby('n_cells')
    summary = t.mean()
    summary['size'] = t.size()
    summary['sem_score'] = t['test_score_hit_vs_miss'].sem()
    plt.errorbar(summary.index.values, summary['test_score_hit_vs_miss'],
        yerr=summary['sem_score'],color='blue')
    plt.plot(summary.index.values, summary.test_score_hit_vs_miss,'bo-',
        label='timing sessions')

    # Clean up hit vs miss decoder performance plot
    plt.xlim(0,80)
    plt.ylim(0.5,1)
    plt.ylabel('hit vs miss decoder performance',fontsize=16)
    plt.xlabel('number of cells',fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # save hit vs miss decoder performance figure
    if savefig:
        filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet'+\
            '/behavior/decoding/figures/' 
        filename += cell_type+'_hit_vs_miss_decoder_performance.png'
        plt.savefig(filename)


def plot_by_strategy(results_df,aggregate_first=True,cell_type='exc',
    areas=['VISp','VISl'],equipment=['MESO.1','CAM2P.3','CAM2P.4','CAM2P.5'],
    savefig=False):
   
    # parse cell type 
    mapper = {
        'exc':'Slc17a7-IRES2-Cre',
        'sst':'Sst-IRES-Cre',
        'vip':'Vip-IRES-Cre'
        }
    cre = mapper[cell_type]
    
    # filter out experiments
    results_df = results_df.query('experience_level == "Familiar"')
    results_df = results_df.query('cre_line == @cre')  
    results_df = results_df.query('targeted_structure in @areas')
    results_df = results_df.query('equipment_name in @equipment') 

    # Average over samples from the same experiment, so each experiment 
    # is weighted the same
    if aggregate_first:   
        x = results_df.groupby(['n_cells','ophys_experiment_id']).mean()
        results_df = x.reset_index() 

    # Split by strategy
    visual = results_df.query('visual_strategy_session')
    timing = results_df.query('not visual_strategy_session')


    plot_by_strategy_performance(visual, timing,savefig,cell_type)
    plot_by_strategy_correlation(visual, timing, savefig, cell_type)
    plot_by_strategy_hit_vs_miss(visual, timing, savefig, cell_type)
    plot_by_strategy_scatter(visual, timing,'behavior_correlation', 
        savefig, cell_type,version)
    plot_by_strategy_scatter(visual, timing,'test_score',savefig,
        cell_type,version)


def plot_by_cre_FA(results_df, aggregate_first=True,areas=['VISp','VISl'],
    equipment=['MESO.1','CAM2P.3','CAM2P.4','CAM2P.5'],ncells=2,
    savefig=False,version=None,meso=False):

    # filter out experiments
    results_df = results_df.query('experience_level == "Familiar"')
    results_df = results_df.query('targeted_structure in @areas')
    results_df = results_df.query('equipment_name in @equipment') 
    results_df = results_df.query('n_cells > @ncells')

    # Split by strategy
    cres = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
    mapper = {
        'Slc17a7-IRES2-Cre':'exc',
        'Sst-IRES-Cre':'sst',
        'Vip-IRES-Cre':'vip'
        }
    fig1, ax1 = plt.subplots()
    stats = []
    for index,cre in enumerate(cres):
        cre_df = results_df.query('cre_line == @cre')
        # Average over samples from the same experiment, so each experiment 
        # is weighted the same
        if aggregate_first:   
            x = cre_df.groupby(['n_cells','ophys_experiment_id']).mean()
            cre_df = x.reset_index() 

        visual = cre_df.query('(visual_strategy_session)').copy()
        timing = cre_df.query('(not visual_strategy_session)').copy()
        if version == 8:
            timing['test_score_FA'] = timing['test_score_FA2']
        if index == 2:
            plt.figure(fig1.number)
            df = plot_by_strategy_scatter(visual,timing,'test_score_FA',
                savefig,mapper[cre],version,ax1,meso=meso)
        else:
            df = plot_by_strategy_scatter(visual,timing,'test_score_FA',
                False,mapper[cre],version,ax1,meso=meso)
        stats.append(df)
    return stats

def plot_by_cre(results_df, aggregate_first=True,areas=['VISp','VISl'],
    equipment=['MESO.1','CAM2P.3','CAM2P.4','CAM2P.5'],ncells=2,
    savefig=False,version=None,meso=False):

    # filter out experiments
    results_df = results_df.query('experience_level == "Familiar"')
    results_df = results_df.query('targeted_structure in @areas')
    results_df = results_df.query('equipment_name in @equipment') 
    results_df = results_df.query('n_cells > @ncells')

    # Split by strategy
    cres = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
    mapper = {
        'Slc17a7-IRES2-Cre':'exc',
        'Sst-IRES-Cre':'sst',
        'Vip-IRES-Cre':'vip'
        }
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    for index,cre in enumerate(cres):
        cre_df = results_df.query('cre_line == @cre')
        # Average over samples from the same experiment, so each experiment 
        # is weighted the same
        if aggregate_first:   
            x = cre_df.groupby(['n_cells','ophys_experiment_id']).mean()
            cre_df = x.reset_index() 

        visual = cre_df.query('(visual_strategy_session)')
        timing = cre_df.query('(not visual_strategy_session)')
        if index == 2:
            plt.figure(fig1.number)
            plot_by_strategy_scatter(visual,timing,'behavior_correlation',
                savefig,mapper[cre],version,ax1,meso=meso)
            plt.figure(fig2.number)
            plot_by_strategy_scatter(visual,timing,'test_score',
                savefig,mapper[cre],version,ax2,meso=meso)
            plt.figure(fig3.number)
            plot_by_strategy_scatter(visual,timing,'test_score_hit_vs_miss',
                savefig,mapper[cre],version,ax3,meso=meso)
        else:
            plot_by_strategy_scatter(visual,timing,'behavior_correlation',
                False,mapper[cre],version,ax1,meso=meso)
            plot_by_strategy_scatter(visual,timing,'test_score',
                False,mapper[cre],version,ax2,meso=meso)
            plot_by_strategy_scatter(visual,timing,'test_score_hit_vs_miss',
                False,mapper[cre],version,ax3,meso=meso)

def stats_by_strategy(cre,visual, timing, test='test_score'):
    n_cells = set(visual.n_cells.unique()).intersection(set(timing.n_cells.unique()))
    n_cells = np.sort(list(n_cells)) 
    tests = []
    for n in n_cells:
        output = stats.ttest_ind(
            visual.query('n_cells==@n')[test],
            timing.query('n_cells==@n')[test])
        tests.append({
            'n_cells':n,
            'cre':cre,
            'test':test,
            'p':output.pvalue
            })
    df = pd.DataFrame(tests)
    return df

def iterate_n_cells(cells,FA=False):
    n_cells = [1,2,5,10,20,40,80]
 
    results = {} 
    for n in n_cells:
        if len(cells) < n:
            break
        print('Decoding with n={} cells'.format(n))
        results[n] = decode_cells(cells, n,FA=FA)

    results_df = pd.concat(results)
    return results_df


def decode_cells(cells, n_cells,FA=False):
    '''
        Cells is a list of dataframes, one for each cell
        n_cells is the number of cells to decode with in each sample
    '''
    
    # How many times we need to sample in order to get 99% chance each cell 
    # is used at least once
    if n_cells == 1:
        n_samples = len(cells)
    else:
        n_samples = int(np.ceil(np.log(0.01)/np.log(1-n_cells/len(cells))))
    
    # Iterate over samples and save output
    output = []
    for n in tqdm(range(0,n_samples)):
        if n_cells == 1:
            temp = decode_cells_sample(cells, n_cells, index=n,FA=FA)
        else:
            temp = decode_cells_sample(cells, n_cells,FA=FA)
        output.append(temp)

    # Return the output of the samples
    output_df = pd.DataFrame(output)
    output_df['n_cells'] = n_cells
    return output_df


def decode_cells_sample(cells, n_cells,index=None,FA=False):
    '''
        Sample from the list of cells, and perform decoding once
        returns the output of the decoding
    '''
 
    # Sample n_cells from list of cells
    if n_cells == 1:
        sample_cells = [cells[index]]
    else:
        cells_in_sample = np.random.choice(len(cells),n_cells, replace=False)
        sample_cells = [cells[i] for i in cells_in_sample] 
    
    if FA:
        X = []
        for cell in sample_cells:
            X.append(get_matrix(cell.query('FA or pre_FA')))
        X = np.concatenate(X,axis=1)
        y = sample_cells[0].query('FA or pre_FA')['FA'].astype(bool).values

        model = {}
        rfc = RandomForestClassifier(class_weight='balanced')
        model['cv_prediction_FA'] = cross_val_predict(rfc, X,y,cv=5)
        model['test_score_FA'] = np.mean(y == model['cv_prediction_FA'])

        # Do decoding on half the dataset, with respect to number of FA
        index = np.arange(len(y))
        if sample_cells[0].iloc[0]['FA']:
            index = index[1:]
        # check to make sure we have even samples
        if np.mod(len(index),2) != 0:
            raise Exception('should have even samples')
        index = index[0:int(np.floor(len(index)/2))]
        np.random.shuffle(index)
        choice = index[0:int(np.floor(len(index)/2))]
        choice = np.concatenate([[x*2, x*2-1] for x in choice])
        X2 = X[choice,:]
        y2 = y[choice]
        rfc = RandomForestClassifier(class_weight='balanced')
        model['cv_prediction_FA2'] = cross_val_predict(rfc, X2,y2,cv=5)
        model['test_score_FA2'] = np.mean(y2 == model['cv_prediction_FA2'])


    else:
        # Construct X 
        X = []
        for cell in sample_cells:
            X.append(get_matrix(cell))
        X = np.concatenate(X,axis=1)
 
        # y is the same for every cell, since its a behavioral output
        y = sample_cells[0]['is_change'].values

        # run CV decoder: change vs non-change
        model = {}
        rfc = RandomForestClassifier(class_weight='balanced')
        model['cv_prediction'] = cross_val_predict(rfc, X,y,cv=5)
        model['behavior_correlation'] = compute_behavior_correlation(\
            sample_cells[0].copy(),model) 
        model['test_score'] = np.mean(y == model['cv_prediction'])
    
        # run CV decoder: hit vs miss
        X = []
        for cell in sample_cells:
            X.append(get_matrix(cell.query('is_change')))
        X = np.concatenate(X,axis=1)
        y = sample_cells[0].query('is_change')['hit'].astype(bool).values

        rfc = RandomForestClassifier(class_weight='balanced')
        model['cv_prediction_hit_vs_miss'] = cross_val_predict(rfc,X,y,cv=5)
        model['test_score_hit_vs_miss'] = np.mean(y == model['cv_prediction_hit_vs_miss'])   

    # return results
    return model 
 

def compute_behavior_correlation(cell, model):
    cell['prediction'] = model['cv_prediction']   
    cell = cell.query('is_change').copy()
    cell['hit'] = cell['hit'].astype(bool)
    return cell[['prediction','hit']].corr()['hit']['prediction']


def plot_behavior_ratio(df,ncells=2,savefig=False,version=None):

    df = df.query('cre_line == "Slc17a7-IRES2-Cre"')
    df = df.query('experience_level == "Familiar"')
    df = df.query('n_cells > @ncells')

    x = df.groupby(['n_cells','ophys_experiment_id']).mean().reset_index()
    summary = x.groupby(['n_cells','visual_strategy_session'])['behavior_correlation'].mean()
    summary = summary.unstack()
    summary['ratio'] = summary[True]/summary[False]

    plt.figure(figsize=(5,4))
    plt.plot(summary.index.values, summary.ratio.values, 'ko-')
    plt.xlabel('# of cells',fontsize=16)
    plt.ylabel('correlation with behavior \n (ratio of visual/timing)',fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_ylim(0,4)
    ax.set_xlim(0,85)
    plt.tight_layout()

    # save decoder performance figure
    if savefig:
        filename='/allen/programs/braintv/workgroups/nc-ophys/alex.piet'+\
            '/behavior/decoding/figures_fit_{}/'.format(version) 
        filename += 'exc_decoder_correlation_ratio.png'
        print(filename)
        plt.savefig(filename)


















     
