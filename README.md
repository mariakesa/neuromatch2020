# neuromatch2020

"The mystery of disappearing receptive fields”

Maria Kesa

How good is our understanding and our models of synaptic plasticity? We combine experimental data from V1 and simulations to study the effect of synaptic plasticity on the receptive field of an artificial neuron (Stringer et al, 2019). We use a clustering algorithm called Ensemble Pursuit (Kesa, Stringer, Pachitariu, 2019) to extract two neural clusters from data, one localized ensemble associated with neurons that share a similar receptive field and one spatially spread out ensemble whose activity is well predicted by behavior (motion SVD’s of the mouse’s snout) and use neurons from the clusters as synaptic inputs to an artificial neuron.  Without plasticity at the synapses the artificial neuron has a linear receptive field. However when we endow synapses with Bienenstock-Munro-Cooper or Hebbian plasiticity, the receptive fields disappear. Future work will consider developing smart learning rules that permit selective integration of information without disrupting the functionality of neurons. 

References:

Stringer et al, 
“Recording of 19,000 neurons across mouse visual cortex during sparse noise stimuli”, Figshare, 2019

Kesa, Stringer, Pachitariu, “Ensemble Pursuit: an algorithm for finding overlapping clusters of correlated neurons in large-scale recordings”, Carnegie Mellon Statistical Analysis of Neural Data poster, 2019
