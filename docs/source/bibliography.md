# Bibliography

This bibliography contains sources actually used by the CELNN documentation. A citation establishes only the claim for which it is cited; it does not imply that the library reproduces every assumption, circuit constraint, or algorithm in the source.

(chua-yang-1988-theory)=
## Chua and Yang — Cellular Neural Networks: Theory

L. O. Chua and L. Yang, “Cellular Neural Networks: Theory,” *IEEE Transactions on Circuits and Systems*, vol. 35, no. 10, pp. 1257–1272, October 1988.

The foundational paper defines the cellular architecture, local neighborhoods, state/input/output variables, feedback and control operators, continuous-time dynamics, and stability results for the circuit family studied there.

(chua-yang-1988-applications)=
## Chua and Yang — Cellular Neural Networks: Applications

L. O. Chua and L. Yang, “Cellular Neural Networks: Applications,” *IEEE Transactions on Circuits and Systems*, vol. 35, no. 10, pp. 1273–1290, October 1988.

The companion paper develops image-processing and pattern-recognition examples and emphasizes the role of both steady-state and transient behavior.

(crounse-chua-1995)=
## Crounse and Chua — Image processing and pattern formation tutorial

K. R. Crounse and L. O. Chua, “Methods for Image Processing and Pattern Formation in Cellular Neural Networks: A Tutorial,” *IEEE Transactions on Circuits and Systems I: Fundamental Theory and Applications*, vol. 42, no. 10, pp. 583–601, October 1995.

This tutorial analyzes linearized CNN dynamics in the spatial-frequency domain and relates standard templates to filtering, diffusion-like behavior, and pattern formation.

(turing-1952)=
## Turing — The Chemical Basis of Morphogenesis

A. M. Turing, “The Chemical Basis of Morphogenesis,” *Philosophical Transactions of the Royal Society of London. Series B, Biological Sciences*, vol. 237, pp. 37–72, 1952.

This is used only for the reaction–diffusion idea that diffusion coupled to local reaction dynamics can destabilize a homogeneous state and produce spatial structure. It is not evidence that an arbitrary CELNN template is a Turing system.

(kozek-roska-chua-1993)=
## Kozek, Roska, and Chua — Genetic algorithm for CNN template learning

T. Kozek, T. Roska, and L. O. Chua, “Genetic Algorithm for CNN Template Learning,” *IEEE Transactions on Circuits and Systems I: Fundamental Theory and Applications*, vol. 40, no. 6, pp. 392–402, June 1993.

The paper formulates CNN template learning as optimization and applies a genetic algorithm to derive templates from task performance.

(schuler-et-al-1992)=
## Schuler et al. — Learning state-space trajectories

A. J. Schuler, P. Nachbar, J. A. Nossek, and L. O. Chua, “Learning State Space Trajectories in Cellular Neural Networks,” in *Proceedings of the Second International Workshop on Cellular Neural Networks and Their Applications (CNNA ’92)*, pp. 68–73, 1992.

The paper applies a trajectory-level error functional and calculus of variations to compute gradients in CNN parameter space.

(oja-1982)=
## Oja — A simplified neuron model as a principal component analyzer

E. Oja, “A Simplified Neuron Model as a Principal Component Analyzer,” *Journal of Mathematical Biology*, vol. 15, pp. 267–273, 1982.

Oja derives a normalized Hebbian-type local learning rule and analyzes its principal-component behavior.

(kohonen-1972)=
## Kohonen — Correlation matrix memories

T. Kohonen, “Correlation Matrix Memories,” *IEEE Transactions on Computers*, vol. C-21, pp. 353–359, 1972.

Kohonen describes key/data associative recall using a correlation matrix whose entries accumulate products of key and data components.

(schmidhuber-1992)=
## Schmidhuber — Learning to control fast-weight memories

J. Schmidhuber, “Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks,” *Neural Computation*, vol. 4, pp. 131–139, 1992.

The paper treats rapidly changing weights as short-term memory controlled by a slower learning system and discusses temporary associations and variable binding.

(katharopoulos-et-al-2020)=
## Katharopoulos et al. — Linear attention

A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, “Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention,” in *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, pp. 5156–5165, 2020.

The paper rewrites kernelized attention using recurrent accumulators and uses the positive feature map `elu(x) + 1` in its experiments. CELNN’s normalized associative field is its own implementation and derivation; this citation supports only the feature-map/normalizer lineage described in the explanation pages.

## Attribution boundary

The current source set does not include the primary Widrow and Hoff 1960 paper. Consequently, this documentation does **not** use an indirect citation to claim historical provenance of CELNN’s `DeltaHebbianRule`. The implemented update is documented from the implementation itself; a historical attribution can be added later only after the primary source is verified.
