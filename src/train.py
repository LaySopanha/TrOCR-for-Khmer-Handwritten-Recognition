


data_dir, df = create_dummy_dataset()

print("Loading processor...")
feature_extractor_name = "microsoft/trocr-small-handwritten"
tokenizer_name = "xlm-roberta-base"

processor = TrOCRProcessor.from_pretrained(feature_extractor_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
processor.tokenizer = tokenizer

dataset = KhmerOCRDataset(root_dir = data_dir, df=df, processor=processor)

print("Loading Model....")
model = VisionEncoderDecoderModel.from_pretrained(feature_extractor_name)

model.decoder.resize_token_embeddings(len(processor.tokenizer))
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = len(processor.tokenizer)

model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

training_args = Seq2SeqTrainingArguments(
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    num_train_epochs = 5,
    predict_with_generate = True,
    logging_steps = 2,
    save_steps = 100,
    eval_strategy = "no",
    fp16= torch.cuda.is_available(),
    remove_unused_columns = False,
)

trainer = Seq2SeqTrainer(
    model = model,
    tokenizer = processor.feature_extractor,
    args = training_args,
    train_dataset = dataset,
    data_collator = default_data_collator,
)

print("Starting Training....")
trainer.train()

print("Training Finished! Saving Model...")
model.save_pretrained("./khmer_trocr_model")
processor.save_pretrained("./khmer_trocr_model")

print("\n--- Running Inference Test ---")
image = Image.open(os.path.join(data_dir, "img_3.jpg")).convert("RGB")
pixel_values = processor(image, return_tensors = "pt").pixel_values.to(model.device)

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Original Text: កម្ពុជា")
print(f"Predicted Text: {generated_text}")
