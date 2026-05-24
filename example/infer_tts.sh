export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"

# Note: To infer Chinese dialects, set model_dir to "pretrained_models/SoulX-Podcast-1.7B-dialect 
#       and follow example/podcast_script/script_mandarin to set dialect_prompt and pass it to 
#       entrance tts.py.
# Example: infer chinese sichuan dialects with mandarin prompt:
# dialect_prompt="<|Sichuan|>要得要得！前头几个耍洋盘，我后脚就背起铺盖卷去景德镇耍泥巴，巴适得喊老天爷！"
# text="<|Sichuan|>一个着迷于在岩壁与雪原间捕捉语言灵感的旅人，即将奔赴景德镇将朝露炊烟揉进陶胚的造梦者。"

model_dir=pretrained_models/SoulX-Podcast-1.7B
prompt_text="喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。"
dialect_prompt=""
prompt_audio="example/audios/female_mandarin.wav"
text="一个着迷于在岩壁与雪原间捕捉语言灵感的旅人，即将奔赴景德镇将朝露炊烟揉进陶胚的造梦者。"

python cli/tts.py \
        --prompt_text ${prompt_text} \
        --dialect_prompt "${dialect_prompt:-}"  \
        --prompt_audio ${prompt_audio} \
        --text ${text} \
        --model_path ${model_dir} \
        --output_path outputs/mandarin_tts.wav \
        --seed 7