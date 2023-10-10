import psy_output_tools as po
import visual_behavior_glm_strategy.PSTH as psth
import visual_behavior_glm_strategy.GLM_params as glm_params
import visual_behavior_glm_strategy.GLM_fit_tools as gft
import visual_behavior_glm_strategy.GLM_fit_dev as gfd
import visual_behavior_glm_strategy.decoding as d
import visual_behavior_glm_strategy.GLM_schematic_plots as gsm
import visual_behavior_glm_strategy.GLM_perturbation_tools as gpt
import visual_behavior_glm_strategy.GLM_strategy_tools as gst
import matplotlib.pyplot as plt
plt.ion()
from importlib import reload
from alex_utils import *

BEHAVIOR_VERSION = 21
summary_df  = po.get_ophys_summary_table(BEHAVIOR_VERSION)


## Fig. 4C - Example ophys schematic
################################################################################
experiment_table = glm_params.get_experiment_table()
oeid = 956903412 
cell_id = 1086505751
GLM_VERSION = '24_events_all_L2_optimize_by_session'
run_params = glm_params.load_run_json(GLM_VERSION)
session = gft.load_data(oeid, run_params)
time=[1221.25, 1225.75]
gsm.strategy_paper_ophys_example(session, cell_id, time)


## Fig. 4D - Population average response
################################################################################
dfs = psth.get_figure_4_psth(data='events',mesoscope_only=True)
psth.plot_figure_4_averages(dfs, data='events',meso=True)

# Determine significant for Vip image
vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events',
    first=False, second=True,meso=True)
psth.plot_summary_bootstrap_image_strategy(vip_image, 'vip',
    first=False, second=True,meso=True)

# Determine significance for SST omission
sst_omission = psth.load_omission_df(summary_df,cre='Sst-IRES-Cre',data='events',
    first=False, second=True,meso=True)
psth.plot_summary_bootstrap_omission_strategy(sst_omission,'sst',first=False,
    second=True,meso=True)

# Determine significance for VIP omission
vip_omission = psth.load_omission_df(summary_df,cre='Vip-IRES-Cre',data='events',
    second=False, first=False,meso=True)
psth.plot_summary_bootstrap_omission_strategy(vip_omission,'vip',first=False,
    second=False,meso=True)

# Determine significance for Exc hit/miss
exc_change = psth.load_change_df(summary_df, cre='Slc17a7-IRES2-Cre',data='events',
    first=False, second=False, image=True,meso=True)
psth.plot_summary_bootstrap_strategy_hit(exc_change, 'exc', first=False, second=False,
    image=True,meso=True)

# Determine significance for Sst hit/miss
sst_change = psth.load_change_df(summary_df, cre='Sst-IRES-Cre',data='events',
   first=False, second=True,meso=True)
psth.plot_summary_bootstrap_strategy_hit(sst_change,'sst',first=False, second=True,
    meso=True)

# determine pre-change for Vip 
vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events',
    first=False, second=True,meso=True)
psth.plot_summary_bootstrap_strategy_pre_change(vip_image,'vip',first=False, 
    second=True,meso=True)

# Determine exc post-omission by strategy
exc_image = psth.load_image_df(summary_df, cre='Slc17a7-IRES2-Cre',data='events',
    first=False, second=False, image=True, meso=True)
exc_post=exc_image.query('post_omitted_1').copy()
psth.plot_summary_bootstrap_omission_strategy(exc_post, 'exc',first=False, second=False,
    image=True, meso=True,post=True) # no strategy difference after omissions

# determine sst pre/post omission
exc_pre_post = exc_image.query('(pre_omitted_1) or (post_omitted_1)').copy()
psth.plot_summary_bootstrap_pre_post_omission(exc_pre_post, 'exc',first=False, second=False,
    image=True, meso=True) 
psth.plot_summary_bootstrap_pre_post_omission(exc_pre_post.query('visual_strategy_session'), 
    'vexc',first=False, second=False,image=True, meso=True) 
psth.plot_summary_bootstrap_pre_post_omission(exc_pre_post.query('not visual_strategy_session'), 
    'texc',first=False, second=False,image=True, meso=True) 

# Determine sst pre/post omission
sst_image = psth.load_image_df(summary_df, cre='Sst-IRES-Cre',data='events',
    first=False, second=False, meso=True,image=True)
sst_pre_post = sst_image.query('(pre_omitted_1) or (post_omitted_1)').copy()
psth.plot_summary_bootstrap_pre_post_omission(sst_pre_post,'sst',first=False, second=False,
    image=True, meso=True) 
psth.plot_summary_bootstrap_pre_post_omission(sst_pre_post.query('visual_strategy_session'),
    'vsst',first=False, second=False,image=True, meso=True) 
psth.plot_summary_bootstrap_pre_post_omission(sst_pre_post.query('not visual_strategy_session'),
    'tsst',first=False, second=False,image=True, meso=True) 

# Check multiple comparisons
tests = psth.bootstrap_summary_multiple_comparisons()

## Fig. 4F/G/H - Multi-level regression
################################################################################
multilevel_summary = \
    pd.read_csv('/home/alex.piet/codebase/behavior/PSTH/vip_summary_multilevel_regression.csv')

## Fig. 4F - Running VIP control image
################################################################################

vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events',
    meso=True,second=True)
bootstraps_image = psth.get_running_bootstraps('vip','image','events',10000,
    second=True, meso=True)
psth.running_responses(vip_image, 'image',bootstraps=bootstraps_image,meso=True)


## Fig. 4G - Running VIP control Omission
################################################################################

vip_omission = psth.load_omission_df(summary_df, cre='Vip-IRES-Cre',data='events',
    meso=True)
bootstraps_omission = psth.get_running_bootstraps('vip','omission','events',10000,
    meso=True)
psth.running_responses(vip_omission, 'omission',bootstraps=bootstraps_omission,
    meso=True)

## Fig. 4H - Running VIP control hit/miss
################################################################################

vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events',
    meso=True,second=True)
bootstraps=psth.compute_pre_change_running_bootstrap_compare_strategy(vip_image,
    'pre_change','vip')
psth.pre_change_running_responses_compare_strategy(vip_image,'pre_change',change_type='hit',
    bootstraps=bootstraps)


## Fig 5
################################################################################

# Just need for filepaths
GLM_VERSION = '24_events_all_L2_optimize_by_session'
run_params = glm_params.load_run_json(GLM_VERSION)
labels = ['Excitatory','Sst Inhibitory','Vip Inhibitory']

dfs = psth.get_figure_4_psth(data='events',mesoscope_only=True)

# Plot PSTH over time
gpt.PSTH_analysis(dfs,  'image',run_params,meso=True)
gpt.PSTH_analysis(dfs,  'omission',run_params,meso=True)
gpt.PSTH_analysis(dfs,  'hit',run_params,meso=True)
gpt.PSTH_analysis(dfs,  'miss',run_params,meso=True)
gpt.PSTH_analysis(dfs,  'image_fa',run_params,meso=True)

# Plot state space plots
gpt.plot_PSTH_perturbation(dfs,labels,'image',run_params,meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'omission',run_params,meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'hit',run_params,meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'miss',run_params,meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'image_fa',run_params,meso=True,savefig=True)

# Plot 3D state space plots
gpt.plot_PSTH_3D(dfs,labels,'image',run_params,meso=True,savefig=True)

# Supplemental figures
gpt.plot_PSTH_perturbation(dfs,labels,'image',run_params,x='Sst',meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'omission',run_params,x='Sst',meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'hit',run_params,x='Sst',meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'miss',run_params,x='Sst',meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'image_fa',run_params,x='Sst',meso=True,savefig=True)


gpt.plot_PSTH_perturbation(dfs,labels,'image',run_params,y='Sst',meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'omission',run_params,y='Sst',meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'hit',run_params,y='Sst',meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'miss',run_params,y='Sst',meso=True,savefig=True)
gpt.plot_PSTH_perturbation(dfs,labels,'image_fa',run_params,y='Sst',meso=True,savefig=True)


gpt.plot_PSTH_3D(dfs,labels,'image',run_params,supp_fig=True,meso=True,savefig=True)
gpt.plot_PSTH_3D(dfs,labels,'omission',run_params,supp_fig=True,meso=True,savefig=True)
gpt.plot_PSTH_3D(dfs,labels,'hit',run_params,supp_fig=True,meso=True,savefig=True)
gpt.plot_PSTH_3D(dfs,labels,'miss',run_params,supp_fig=True,meso=True,savefig=True)
gpt.plot_PSTH_3D(dfs,labels,'image_fa',run_params,supp_fig=True,meso=True,savefig=True)



## Figure 6 - Decoding
################################################################################

summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION) 
experiment_table = glm_params.get_experiment_table() 
df3 = d.load_all(experiment_table, summary_df,version=3,mesoscope_only=True)
d.plot_by_cre(df3,meso=True)


## Fig. 6 Engagement PSTHs
################################################################################
dfs = psth.get_figure_4_psth(data='events',mesoscope_only=True)
psth.plot_engagement(dfs,data='events',meso=True)

exc_change = psth.load_change_df(summary_df, cre='Slc17a7-IRES2-Cre',data='events',
    first=False, second=False,image=True, meso=True)
psth.plot_summary_bootstrap_strategy_engaged_miss(exc_change,cell_type='exc',
    first=False, second=False,nboots=10000,image=True,meso=True)

exc_image = psth.load_image_df(summary_df,cre='Slc17a7-IRES2-Cre',data='events',
    first=False,second=False,image=True,meso=True)
exc_post=exc_image.query('post_omitted_1').copy()
psth.plot_summary_bootstrap_strategy_engaged_omission(exc_post,data='events',nboots=10000,
    cell_type='exc',first=False,second=False,post=True,meso=True,image=True)

## Fig. S6 - Running VIP control image
################################################################################
vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events',
    second=True,meso=True)
boot_image_visual = psth.compute_engagement_running_bootstrap(vip_image,'image',
    'vip','visual',nboots=10000,meso=True,second=True)
boot_image_timing = psth.compute_engagement_running_bootstrap(vip_image,'image',
    'vip','timing',nboots=10000,meso=True,second=True)
psth.engagement_running_responses(vip_image, 'image',vis_boots=boot_image_visual,
    tim_boots=boot_image_timing, plot_list=['visual'],meso=True,second=True)
psth.engagement_running_responses(vip_image, 'image',vis_boots=boot_image_visual,
    tim_boots=boot_image_timing, plot_list=['timing'],meso=True,second=True)


## Fig. S6 - Running VIP control Omission
################################################################################
vip_omission = psth.load_omission_df(summary_df, cre='Vip-IRES-Cre',data='events',
    meso=True)
boot_omission_visual = psth.compute_engagement_running_bootstrap(vip_omission,
    'omission','vip','visual',nboots=10000,meso=True,compute=False)
boot_omission_timing = psth.compute_engagement_running_bootstrap(vip_omission,
    'omission','vip','timing',nboots=10000,meso=True,compute=False)
psth.engagement_running_responses(vip_omission, 'omission',
    vis_boots=boot_omission_visual,
    tim_boots=boot_omission_timing, 
    plot_list=['visual'],meso=True)
psth.engagement_running_responses(vip_omission, 'omission',
    vis_boots=boot_omission_visual,
    tim_boots=boot_omission_timing, 
    plot_list=['timing'],meso=True)


## Novelty Supplement 
################################################################################
dfs_novel = psth.get_figure_4_psth(data='events',
    experience_level='Novel 1',mesoscope_only=True)
psth.plot_figure_4_averages(dfs_novel, data='events',
    experience_level='Novel 1',meso=True)

## Supplement - reward censored PSTHs 
################################################################################
dfs = psth.get_figure_4_psth(data='events',mesoscope_only=True)
ylims = psth.plot_figure_4_averages(dfs, data='events',meso=True)
psth.plot_figure_4_averages_reward(dfs, data='events',meso=True, ylims=ylims)

## Supplement - PSTHs split by area
################################################################################
ylims = [.0167,.0923,.0628]  
psth.plot_figure_4_averages(dfs, data='events',meso=True,areas=['VISp'],in_ylims=ylims) 
psth.plot_figure_4_averages(dfs, data='events',meso=True,areas=['VISl'],in_ylims=ylims) 

## Supplement - behavior averages
################################################################################
dfs_zrunning = psth.get_figure_4_behavioral('running_zscore')
psth.plot_figure_4_behavioral(dfs_zrunning,data='running_zscore')

dfs_zpupil = psth.get_figure_4_behavioral('pupil_zscore')
psth.plot_figure_4_behavioral(dfs_zpupil,data='pupil_zscore')

dfs_licks = psth.get_figure_4_behavioral('licks')
psth.plot_figure_4_behavioral(dfs_licks,data='licks')

## Supplement - vip running control
################################################################################
psth.plot_figure_4_averages_vip_matched()

## Supplement - False Alarms
################################################################################
dfs = psth.get_figure_4_psth(data='events',mesoscope_only=True)
ylims = psth.plot_figure_4_averages(dfs, data='events',meso=True)
psth.plot_figure_4_averages_licking(dfs, data='events',meso=True, ylims=ylims)

# determine false alarms for exc
exc_fa = psth.load_false_alarm_df(summary_df, cre='Slc17a7-IRES2-Cre',data='events',
    first=False, second=False, image=True, meso=True)
exc_change = psth.load_change_df(summary_df, cre='Slc17a7-IRES2-Cre',data='events',
    first=False, second=False, image=True,meso=True)
ax,bootstrap = psth.plot_summary_bootstrap_strategy_hit(exc_change, 'exc', first=False, second=False,
    image=True,meso=True,return_data=True)
psth.plot_summary_bootstrap_strategy_false_alarm(exc_fa, 'exc',first=False, second=False,
    image=True, meso=True,ax=ax,bootstrap=bootstrap)

# determine false alarms for sst
sst_fa = psth.load_false_alarm_df(summary_df, cre='Sst-IRES-Cre',data='events',
    first=False, second=True, image=False, meso=True)
sst_change = psth.load_change_df(summary_df, cre='Sst-IRES-Cre',data='events',
   first=False, second=True,meso=True)
ax,bootstrap = psth.plot_summary_bootstrap_strategy_hit(sst_change,'sst',first=False, second=True,
    meso=True,return_data=True)
psth.plot_summary_bootstrap_strategy_false_alarm(sst_fa, 'sst',first=False, second=True,
    image=False, meso=True,ax=ax,bootstrap=bootstrap)

# determine false alarms for vip
vip_fa = psth.load_false_alarm_df(summary_df, cre='Vip-IRES-Cre',data='events',
    first=False, second=True, image=False, meso=True)
vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events',
    first=False, second=True,meso=True)
ax,bootstrap = psth.plot_summary_bootstrap_strategy_pre_change(vip_image,'vip',first=False, 
    second=True,meso=True,return_data=True)
psth.plot_summary_bootstrap_strategy_false_alarm(vip_fa, 'vip',first=False, second=True,
    image=False, meso=True,ax=ax,bootstrap=bootstrap)



