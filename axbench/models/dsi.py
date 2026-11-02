from .model import Model
import torch

class DSI(Model):
    """Distributed Sparse Interventions.

    Training happens in the DSI repo.

    At inference the vectors are added at every decoder layer's self_attn.o_proj output.
    Applied at all non-BOS prompt positions and every generated position.
    """

    def __str__(self):
        return 'DSI'

    def train(self, examples, **kwargs):
        raise NotImplementedError(
            "DSI is trained in the DSI repo.")

    def load(self, dump_dir=None, **kwargs):
        concept_id = kwargs["concept_id"]
        path = f"{dump_dir}/dsi/{concept_id}/steering_vectors.pt"
        interventions = torch.load(path, map_location=self.device)
        self.resid_vectors = interventions["resid_vectors"].to(torch.float32).to(self.device)
        self.dsi_meta = interventions.get("meta", {})

    def to(self, device):
        self.device = device
        if hasattr(self, "resid_vectors"):
            self.resid_vectors = self.resid_vectors.to(device)
        return self

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.model.eval()
        self.tokenizer.padding_side = "left"

        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        bos_id = self.tokenizer.bos_token_id

        layers = self.model.model.layers
        assert len(layers) == self.resid_vectors.shape[0], \
            f"model has {len(layers)} layers, vectors {self.resid_vectors.shape[0]}"

        all_generations = []
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i + batch_size]
            input_strings = batch_examples["input"].tolist()
            factors = torch.tensor(batch_examples["factor"].tolist(),
                                   dtype=torch.float32, device=self.device)
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            # all non-pad positions except BOS
            prefill_mask = (inputs.attention_mask.bool()
                            & (inputs.input_ids != bos_id)).to(torch.float32)

            handles = []

            def make_hook(vec):
                def hook(module, hook_inputs, output):
                    v = vec.to(output.dtype)
                    scaled = factors.to(output.dtype)[:, None, None] * v
                    if output.shape[1] > 1:  # prefill: mask pads and BOS
                        return output + prefill_mask[:, :, None].to(output.dtype) * scaled
                    return output + scaled   # decode: every generated position
                return hook

            try:
                for l, layer in enumerate(layers):
                    handles.append(layer.self_attn.o_proj.register_forward_hook(
                        make_hook(self.resid_vectors[l])))
                generations = self.model.generate(
                    **inputs, max_new_tokens=eval_output_length, do_sample=True,
                    temperature=temperature,
                )
            finally:
                for h in handles:
                    h.remove()

            input_lengths = [len(ids) for ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts

        return {"steered_generation": all_generations}
