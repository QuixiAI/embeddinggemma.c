// Reference fixture generator, built against llama.cpp only (never linked into the server).
#include "clip-impl.h"
#include "mtmd-image.h"
#include "mtmd-audio.h"
#include "gguf.h"
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) return 2; // text GGUF, projector GGUF, vision|audio|image|wave
    const bool audio = !strcmp(argv[3], "audio") || !strcmp(argv[3], "wave");
    const bool wave = !strcmp(argv[3], "wave"), image = !strcmp(argv[3], "image");
    llama_backend_init();
    clip_context_params cp = {};
    cp.image_max_tokens = 280;
    cp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    auto clip = clip_init(argv[2], cp);
    clip_image_f32 img;
    clip_image_f32_batch images;
    images.is_audio = audio;
    if (wave) {
        std::vector<float> samples(3200);
        for (size_t i=0;i<samples.size();++i) samples[i]=(int16_t)(10000*sin(2*3.141592653589793*440*i/16000))/32768.0f;
        mtmd_audio_preprocessor_gemma4a pre(clip.ctx_a); pre.initialize();
        std::vector<mtmd_audio_mel> mel;
        if(!pre.preprocess(samples.data(),samples.size(),mel)) return 5;
        img.set_size({(int)mel[0].n_len,128},false,true); img.cpy_buf(mel[0].data);
    } else if (image) {
        clip_image_u8 raw;raw.set_size({48,48},false);
        std::vector<uint8_t> rgb(48*48*3);
        for(int i=0;i<48*48;++i){rgb[3*i]=220;rgb[3*i+1]=30;rgb[3*i+2]=50;}
        raw.cpy_buf(rgb);
        mtmd_image_preprocessor_dyn_size pre(clip.ctx_v);img=pre.preprocess(raw).entries[0];
    } else if (audio) {
        img.set_size({96, 128}, false, true);
        std::vector<float> data(96 * 128);
        for (int f = 0; f < 128; ++f)
            for (int t = 0; t < 96; ++t) data[f * 96 + t] = sinf((t * 128 + f) * 0.017f);
        img.cpy_buf(data);
    } else {
        img.set_size({96, 96}, false, false);
        std::vector<float> data(96 * 96 * 3);
        for (size_t i = 0; i < data.size(); ++i) data[i] = ((i * 37 + i / 71) % 256) / 255.0f;
        img.cpy_buf(data);
    }
    images.entries.push_back(img);
    const int nt = clip_n_output_tokens(audio ? clip.ctx_a : clip.ctx_v, &img);
    std::vector<float> tower(nt * 512);
    if (!clip_image_batch_encode(audio ? clip.ctx_a : clip.ctx_v, 8, &images, tower)) return 3;

    ggml_context *weights = nullptr;
    gguf_init_params gp = {false, &weights};
    gguf_context *gguf = gguf_init_from_file(argv[1], gp);
    ggml_tensor *emb = ggml_get_tensor(weights, "token_embd.weight");
    std::vector<float> inputs((nt + 4) * 512);
    const int ids[] = {2, audio ? 256000 : 255999, audio ? 258883 : 258882, 1};
    const int positions[] = {0, 1, nt + 2, nt + 3};
    for (int i = 0; i < 4; ++i) {
        float *dest = inputs.data() + positions[i] * 512;
        ggml_get_type_traits(emb->type)->to_float((const char *)emb->data + ids[i] * emb->nb[1], dest, 512);
        for (int z = 0; z < 512; ++z) dest[z] *= sqrtf(512.0f);
    }
    memcpy(inputs.data() + 1024, tower.data(), tower.size() * sizeof(float));
    gguf_free(gguf); ggml_free(weights);
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(argv[1], mp);
    auto params = llama_context_default_params();
    params.n_ctx = 8192; params.n_batch = 8192; params.n_ubatch = 8192;
    params.embeddings = true; params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    params.n_threads = 8; params.n_threads_batch = 8;
    auto *ctx = llama_init_from_model(model, params);
    auto batch = llama_batch_init(nt + 4, 512, 1);
    batch.n_tokens = nt + 4;
    memcpy(batch.embd, inputs.data(), inputs.size() * sizeof(float));
    for (int i = 0; i < nt + 4; ++i) {
        batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch)) return 4;
    const float *out = llama_get_embeddings_seq(ctx, 0);
    double norm = 0; for (int i = 0; i < 768; ++i) norm += out[i] * out[i];
    printf("{\"tower\":[");
    for (size_t i = 0; i < tower.size(); ++i) printf("%s%.9g", i ? "," : "", tower[i]);
    printf("],\"embedding\":[");
    for (int i = 0; i < 768; ++i) printf("%s%.9g", i ? "," : "", out[i] / sqrt(norm));
    puts("]}");
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
    clip_free(clip.ctx_v); clip_free(clip.ctx_a); llama_backend_free();
}
