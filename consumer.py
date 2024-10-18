import pika
import json
import smtplib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

rabbitmq_host = 'localhost'
rabbitmq_queue = 'emailQueue'

connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
channel = connection.channel()
channel.queue_declare(queue=rabbitmq_queue, durable=True)


RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("./models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Função para enviar email (se for legítimo)
def send_email(recipient, subject, content):
    # Configurações SMTP (exemplo usando Gmail)
    smtp_server = 'email-smtp.us-west-2.amazonaws.com'
    smtp_port = 587
    smtp_username = 'AKIAYGV7ROMC66GH7EVF'
    smtp_password = 'BExhATIL1IVztDh6MHN4yLi2/d6B9sOQwQ2ONhh14IDX'

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)

        # Formatar email
        message = f'Subject: {subject}\n\n{content}'
        server.sendmail(smtp_username, recipient, message)
        server.quit()
        print(f"Email enviado para {recipient}")
    except Exception as e:
        print(f"Erro ao enviar o email: {e}")

# Função callback para processar a mensagem do RabbitMQ
def callback(ch, method, properties, body):
    email_message = json.loads(body)
    recipient = email_message['recipient']
    subject = email_message['subject']
    content = email_message['content']

    combined_text = f"Subject: {subject}\n\nContent: {content}"

    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item() * 100

    if predicted_class == 0:  # 0 significa legítimo
        print(GREEN + f"Email é legítimo com confiança de {confidence:.2f}%" + RESET)
        send_email(recipient, subject, content)
    else:
        print(content)
        print(RED + f"Email classificado como SPAM com confiança de {confidence:.2f}%" + RESET)

    ch.basic_ack(delivery_tag=method.delivery_tag)

# Consumir mensagens do RabbitMQ
channel.basic_consume(queue=rabbitmq_queue, on_message_callback=callback)

print('Aguardando por emails para processar...')
channel.start_consuming()