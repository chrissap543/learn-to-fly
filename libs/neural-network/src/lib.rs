use rand::Rng; 
use rand::RngCore; 

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn random(rng: &mut dyn RngCore, input_size: usize, output_size: usize) -> Self {
        let neurons = (0..output_size).map(|_| Neuron::random(rng, input_size)).collect(); 

        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
}

#[derive(Debug)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        let bias = rng.random_range(-1.0..=1.0); 
        let weights = (0..input_size).map(|_| rng.random_range(-1.0..=1.0)).collect(); 

        Self { bias, weights }
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }
}

#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq; 
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn random() {
        // Because we always use the same seed, our `rng` in here will
        // always return the same set of values
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let neuron = Neuron::random(&mut rng, 4);
        assert_relative_eq!(neuron.bias, -0.6255188);

        assert_relative_eq!(
            neuron.weights.as_slice(),
            [0.67383933, 0.81812596, 0.26284885, 0.5238805].as_ref()
        );
    }

    #[test]
    fn propagate_neuron() {
        let neuron = Neuron {
            bias: 0.5,
            weights: vec![-0.3, 0.8],
        };

        // Ensures `.max()` (our ReLU) works:
        assert_relative_eq!(
            neuron.propagate(&[-10.0, -10.0]),
            0.0,
        );

        // `0.5` and `1.0` chosen by a fair dice roll:
        assert_relative_eq!(
            neuron.propagate(&[0.5, 1.0]),
            (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
        );
    }

    #[test]
    fn propagate_layers() {
        let neuron1 = Neuron {
            bias: 0.5,
            weights: vec![-0.3, 0.8],
        };
        let neuron2 = Neuron {
            bias: 0.25,
            weights: vec![-0.7, 0.9],
        };

        let layer = Layer {
            neurons: vec![neuron1, neuron2],
        };

        layer.propagate(vec![-10.0, -10.0]).iter().for_each(|&x| assert_relative_eq!(x, 0.0));

        let vals = layer.propagate(vec![0.7, 0.4]);
        assert_relative_eq!(vals[0], (-0.3 * 0.7) + (0.8 * 0.4) + 0.5); 
        assert_relative_eq!(vals[1], (-0.7 * 0.7) + (0.9 * 0.4) + 0.25); 
    }

    #[test]
    fn propagate_network() {
        let neuron1 = Neuron {
            bias: 0.5,
            weights: vec![-0.3, 0.8],
        };
        let neuron2 = Neuron {
            bias: 0.25,
            weights: vec![-0.7, 0.9],
        };

        let neuron3 = Neuron {
            bias: 0.5,
            weights: vec![-0.5, 0.7],
        };
        let neuron4 = Neuron {
            bias: 0.25,
            weights: vec![-0.1, 0.9],
        };


        let layer1 = Layer {
            neurons: vec![neuron1, neuron2],
        };
        let layer2 = Layer {
            neurons: vec![neuron3, neuron4],
        };

        let network = Network {
            layers: vec![layer1, layer2],
        }; 

        // network.propagate(vec![-10.0, -10.0]).iter().for_each(|&x| assert_relative_eq!(x, 0.0));

        let vals = network.propagate(vec![0.5, 0.7]); 

        let layer1 = [(-0.3 * 0.5) + (0.8 * 0.7) + 0.5, (-0.7 * 0.5) + (0.9 * 0.7) + 0.25];
        let layer2 = [(-0.5 * layer1[0]) + (0.7 * layer1[1]) + 0.5, (-0.1 * layer1[0]) + (0.9 * layer1[1]) + 0.25]; 
        assert_relative_eq!(vals[0], layer2[0]); 
        assert_relative_eq!(vals[1], layer2[1]); 
    }
}