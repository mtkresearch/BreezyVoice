from single_inference import add_spk

if __name__ == "__main__":
    '''
    test command:
    
    python3 add_spk.py --spk_id 臺灣女 \
        --speaker_prompt_audio_path "data/example.wav" \
        --speaker_prompt_text_transcription "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。只有擁有解密方法的對象，經由解密過程，才能將密文還原為正常可讀的內容。"
    '''
    add_spk()
