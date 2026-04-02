

Video files under experiments & old_videos
Delete after 5m

how to monitor lipschitz smoothness. Lipschitz condition means no corners like ReLU.
compare
  weight decay
  gradient clipping
  spectral norm

use spectral norm on 1-2 first layers
use lipschitz to detect bad rollouts
lipschitz correlates with advantage variance

ppo.yaml stores hyperparameter settings
use hydra for experiment logging and hyperparameter tracking
callback method of updating sb3 ppo. Use callback style in libs. 


