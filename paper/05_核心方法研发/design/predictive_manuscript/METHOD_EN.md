# Methods

## Problem and procedure

We study client-specific probability prediction after federated training with compressed uploads. Client $i\in\{1,\ldots,M\}$ has a local training set $D_i$, a disjoint development holdout $H_i$, and $C$ classes. A shared network produces logits $z(x;w)\in\mathbb{R}^{C}$. The procedure combines count-adjusted local training, block quantization with error feedback, and a local affine transformation fitted after freezing the final shared network. The primary objective is lower client-macro holdout negative log likelihood (NLL), subject to predefined accuracy-preservation constraints. Authenticated upload handling is evaluated separately in a local durable protocol simulation. These components use established constructions; their combination does not assert atomic methodological novelty.

## Data access and federated training

All training counts, priors, gradients, and calibration fits use $D_i$ exclusively. CIFAR-10 loading reads only the five official TRAIN files; FashionMNIST loading reads only its two TRAIN gzip files. File identities, label order, sample identifiers, and memberships are recorded. Classwise Dirichlet-multinomial allocation uses $M=20$, $C=10$, and concentration $0.1$, retrying until every client has at least 100 assigned examples. Each assignment is shuffled and divided approximately 75:25 into $D_i$ and $H_i$. A fixed stratified 2,000-example monitoring subset comes from the holdouts. Only the final checkpoint is used for the comparisons; monitoring does not select a checkpoint. Official TEST files are excluded, so these holdouts support development evaluation.

Both tasks use the same 136,874-parameter SmallCNN. Fashion images are zero-padded from $28\times28$ to $32\times32$, repeated across three channels, and normalized as $(x/255-0.5)/0.5$, matching the CIFAR normalization. Training uses 20 full-participation rounds, one local epoch, batch size 64, and SGD with learning rate $0.03$. Paired methods share initialization and batch seeds. Let $n_{ic}=\sum_{(x,y)\in D_i}\mathbf{1}[y=c]$. Our explicitly specified FedLC-inspired loss is

$$
\ell_i(z,y)=\operatorname{CE}(z-s_i,y),\qquad s_{ic}=\max(n_{ic},1)^{-1/4}.
$$

The count floor and unit multiplier are implementation choices. We use ordinary cross-entropy over all classes and do not claim an exact reproduction of FedLC \cite{zhang2022fedlc}. These adjusted logits are used during training; post-training adaptation receives the original network logits.

## Encoded updates and error feedback

At round $t$, client $i$ forms its local model difference $\Delta_i^t$. With committed residual $r_i^t$, the compressed update and aggregation are

$$
\begin{aligned}
u_i^t&=\Delta_i^t+r_i^t,\quad q_i^t=\operatorname{Decode}(\operatorname{Encode}(u_i^t)),\\
r_i^{t+1}&=u_i^t-q_i^t,\quad w^{t+1}=w^t+\sum_i\frac{|D_i|}{\sum_j|D_j|}q_i^t.
\end{aligned}
$$

Residuals and reconstructed updates use FP32. This is standard error feedback \cite{karimireddy2019errorfeedback}. The synchronous learning implementation updates residuals after decoding; the receipt-dependent durable variant below is a separate protocol simulation.

Uniform3 divides the flattened update into blocks of at most 256 coordinates. Each block stores a one-byte bit-width indicator set to 3 and an FP32 scale, followed by packed three-bit codes representing integers $-3,\ldots,3$. Scale fitting evaluates maximum-magnitude initialization and four least-squares/reassignment refinements, retaining the candidate with least decoded squared error. Every candidate uses its serialized FP32 scale. A 64-byte header records the version and layout; strict decoding checks lengths, metadata, scales, codes, padding, and trailing bytes. For 136,874 coordinates, the actual payload is 54,067 B, below the 54,808 B cap. Aggregation uses decoded wire bytes rather than encoder-side tensors. Dense uploads and all dense downlinks are accounted analytically; no network latency is inferred.

## Local prior-regularized vector adaptation

After freezing $w$, one additional TRAIN forward pass saves each client's logits and labels. Add-one smoothing defines

$$
\widehat\pi_{ic}=\frac{n_{ic}+1}{|D_i|+C},\qquad
\mu_{ic}=\log\widehat\pi_{ic}-\frac{1}{C}\sum_{k=1}^{C}\log\widehat\pi_{ik}.
$$

Thus $\mu_i$ also equals centered $\log(n_i+1)$. It supplies a local regularization anchor, without assuming that a prior shift exactly describes the network's errors. Client $i$ fits

$$
\begin{aligned}
\min_{A_i,b_i}\;&\sum_{(x,y)\in D_i}\operatorname{CE}(A_i\odot z(x;w)+b_i,y)\\
&+\frac{10}{2}\|A_i-\mathbf{1}\|_2^2+\frac{10}{2}\|b_i-\mu_i\|_2^2,\\
\text{subject to }&A_i\in\mathbb{R}^{C},\quad\mathbf{1}^{\mathsf T}b_i=0.
\end{aligned}
$$

The loss is summed, not averaged, so the effective influence of the fixed penalty depends on local sample count. Class-specific scales may be negative. Centering removes the common-logit translation freedom. With fixed logits, affine cross-entropy and positive quadratic penalties give a strictly convex fitting problem on this constraint set; this observation does not provide a federated convergence claim. Vector scaling itself is established \cite{guo2017calibration}.

The fixed solver starts from $(\mathbf{1},\mu_i)$ and uses FP64 analytic gradients and Hessians, at most 50 Newton steps, Armijo coefficient $10^{-4}$, at most 40 backtracking halvings, a projected-gradient fallback, and gradient tolerance $10^{-8}$. It projects the bias to zero mean. The deployed $A_i,b_i$ are rounded once to FP32. All reported scores apply these stored values to the original logits in FP64, consistently across calibration controls, before softmax and argmax. Accuracy preservation is measured because this transformation can change class predictions.

## Endpoints and fixed comparison rules

For probabilities $p_i(x)$, client NLL is $-|H_i|^{-1}\sum_{(x,y)\in H_i}\log p_{iy}(x)$; the primary endpoint averages it equally over clients. Accuracy uses the same client-macro aggregation. We also report sample-weighted metrics, the tenth percentile of client accuracy, multiclass Brier score $|H_i|^{-1}\sum_{(x,y)\in H_i}\sum_c(p_{ic}(x)-\mathbf{1}[y=c])^2$, and top-label ECE with 15 equal-width confidence bins. Bins are left-closed and right-open, with confidence one included in the last bin. Empty bins contribute zero. Mean client ECE and pooled-bin ECE are reported distinctly. Lower NLL does not imply lower Brier or ECE.

The fixed development rule requires NLL reductions of at least $0.01$ against both same-model zero-anchor vector scaling and $z+\log\widehat\pi_i$ prediction, with accuracy losses at most $0.5$ and $0.2$ percentage points, respectively. Against prior-anchor bias alone, NLL may increase by at most $0.005$ and accuracy fall by at most $0.5$ points. Against CE-uniform3 or FedLC-dense with the same vector adaptation, respective limits are $0.01$ NLL and $0.5$ accuracy points. Passing the CE comparison does not establish that count-adjusted training is necessary.

Each task uses exactly three predefined training seeds on one fixed partition: CIFAR-10 partition 20261217 with seeds 20261231--20261233, and FashionMNIST partition 20261219 with seeds 20261241--20261243. Extension beyond the first seed requires all first-seed conditions. Three-seed summaries report individual values and paired means with sample standard deviations; task-level support additionally requires every seed to preserve the specified accuracies, positive NLL reductions against the first two controls in every seed, and mean reductions of at least $0.01$. Clients and rounds are not independent training replicates. No seed, round, penalty, or calibration family is selected by holdout performance.

## Controls and attribution

Deleting $\mu_i$ while retaining twenty stored parameters isolates the prior anchor. Fixing $A_i=\mathbf{1}$ gives prior-anchor bias adaptation. Raw logits, prior-only probabilities $\widehat\pi_i$, and $z+\log\widehat\pi_i$ distinguish adaptation from using label frequencies alone. Positive scalar temperature scaling and bias-corrected temperature scaling (BCTS), $az+b$ with $a\ge0.001$, examine simpler score transformations \cite{guo2017calibration,alexandari2020labelshift}. BCTS is fitted with both zero and the same prior bias anchor. All numerical fits use TRAIN data and coefficient 10; equal coefficients across different dimensions do not equalize statistical complexity. A near-matching BCTS result would not support the necessity of class-specific scales. A matched one-epoch final-Linear control freezes the feature extractor and adapts 650 parameters, with and without the fixed prior.

CE-uniform3 and FedLC-dense address training-loss and compression attribution. The same-cap TopK+EF reference retains 6,843 largest-magnitude coordinates, resolves ties by lower index, and serializes ascending uint32/FP32 pairs with a 64-byte header. Its actual 54,808 B payload exceeds uniform3 by 741 B; equal caps do not mean equal traffic. A FedRoD-linear reference trains a generic branch with balanced softmax using unsmoothed TRAIN counts and a persistent private residual head with ordinary cross-entropy. Its personalized score is generic plus residual logits; the personal loss does not update the shared feature extractor or generic head \cite{chen2022fedrod}. We adopt the final global-generic-plus-local-residual interpretation in its Appendix B.2 and disclose this implementation choice. This reference examines persistent personalization, rather than merely extra post-training affine capacity. These controls define attribution questions without presuming their outcomes.

## Durable upload handling and trust

The separate local protocol binds the full compressed payload to activity, client, round, record identifier, nonce, base-model hash, parameter layout, codec name, and codec-specification SHA256. Canonical length-prefixed serialization rejects malformed or trailing bytes. HMAC-SM3 authenticates the complete envelope; its SM3 digest identifies the recorded bytes. HMAC uses a shared secret \cite{rfc2104}; the trusted server holds each client's key. Specification and source hashes identify intended semantics, while correct decoder execution remains a trust assumption.

A client durably stores one pending envelope and its candidate residual before transmission. Retries query or resend identical bytes without re-encoding. Registration binds identifiers and nonce to bytes; acceptance checks the fixed context and strictly decodes the payload. Once all fixed participants are accepted, one local SQLite transaction commits the weighted model update, accumulator, round result, and incorporated record states. The client commits its residual only after a matching incorporated receipt. Registered, accepted, or unknown status retains the pending state. Repeated finalization and client commit are idempotent.

The full-dimensional integration replays two distinct previously generated payloads with constructed updates, rather than retraining clients through this protocol. Its scope is local simulation of mutation rejection, replay, crash recovery, and acknowledgement loss. There is no deployed blockchain, real network measurement, or distributed atomicity guarantee. The server, database, receipts, and decoder are trusted. A key holder can authenticate a poisoned update: this mechanism is payload authentication, not a weight watermark, public-key signature, poisoning defense, or proof of honest training.

## Cost accounting

For ten classes, deployment adds twenty FP32 calibration values, or 80 B per client; these values stay local. This excludes the shared model, each 547,496 B FP32 error-feedback residual, checkpoints, and runtime allocations. TRAIN logits alone require $40|D_i|$ B in FP32; FP64 copies, affine-design caches, Hessians, and holdout arrays are counted separately. Adaptation requires a TRAIN forward pass and numerical solves, with no additional CNN backward or model upload. The matched frozen-feature head also avoids feature-extractor backpropagation. FedRoD's persistent 650-parameter head requires 2,600 B plus its training work. Codec payload, authenticated-envelope/tag, durable-storage, downlink, forward/cache, and fitting costs have separate accounting scopes. Small calibration state and zero CNN backward do not establish an energy or end-to-end speed advantage.
