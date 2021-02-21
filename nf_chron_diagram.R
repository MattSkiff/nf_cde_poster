# diagram of NF models

library(DiagrammeR)

gr <-grViz("digraph flowchart {
      # node definitions with substituted label text
      node [fontname = Helvetica, shape = rectangle]        
      # edge definitions with the node IDs
      'Linear Combiner (Perceptron) (ANN)' -> 'Multiple Linear Combiner (One Layer Perceptron)';
			'Multiple Linear Combiner (One Layer Perceptron)'-> 'MLP (FFNN)'
			'MLP (FFNN)' -> 'RBF-NN';
			'Neocognitron' -> 'LeNet'-> 'AlexNet (DCNNs)';
			'AlexNet (DCNNs)' -> 'DeepFace'; 
			'AlexNet (DCNNs)' -> 'VGG'; 
			'AlexNet (DCNNs)' -> 'Inception Net'; 
			'AlexNet (DCNNs)' -> 'ResNet';
			'AlexNet (DCNNs)' -> 'R-CNN' -> 'Fast R-CNN';
			'AlexNet (DCNNs)' -> 'YOLO';
			'AlexNet (DCNNs)' -> 'SqueezeNet';
			'LeNet' -> 'GNN' -> 'GTN' -> 'GPT-2';
			'GTN' -> 'BERT';
			'ResNets' -> 'RevNets';
			'ResNets' -> 'iRevNets';
			'ResNets' -> 'iResNets';
			'ResNets' -> 'Residual flow'
			'ResNets' -> 'TDNN';
			'TDNN' -> 'RNNs' -> 'BRNN';
			'RNNs' -> 'LTSM';
			'SOM (CNNs)' -> 'Neocognitron'
			'MLP (FFNN)' -> 'Hopfield Nets' -> 'RNNs';
			'DBM' -> 'GSN'; 
			'Flow++' -> 'Boltzmann Machine' -> 'RBM' -> 'DBM';
			'GSN' -> 'IAF' -> 'GauGAN';
			'MAF' -> 'NICE' -> 'RealNVP';
			'Linear splines' -> 'Monotone Quadratic splines' -> 'Cubic splines' -> 'Rational Quadratic splines'
			'NODE' -> 'FFJORD' 
			'NODE' -> 'ANODE'
			'NODE' -> 'NSDEs'
      }
      ")

gr