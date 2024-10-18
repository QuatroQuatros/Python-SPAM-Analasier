# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# # Códigos ANSI para cores
# RED = '\033[91m'
# GREEN = '\033[92m'
# RESET = '\033[0m'

# # Carregar o modelo e o tokenizer salvos
# tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
# model = AutoModelForSequenceClassification.from_pretrained("./models")

# # Mover o modelo para a GPU (se disponível)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()  # Colocar o modelo em modo de avaliação

# test_emails = [
#     "Reunião de equipe agendada para amanhã às 10h. Favor confirmar presença.",
#     "Parabéns! Você ganhou um prêmio de R$ 1.000,00. Clique aqui para resgatar.",
#     "Fatura de cartão de crédito disponível para visualização. Acesse sua conta para mais detalhes.",
#     "Sua conta foi bloqueada. Para reativar, clique no link e atualize suas informações.",
#     "Convite para o seminário sobre segurança cibernética na próxima semana.",
#     "Você foi selecionado para uma oferta exclusiva! Responda agora para garantir.",
#     "Atualização do software disponível. Clique para baixar a versão mais recente.",
#     "Atenção! Sua conta será desativada em 24 horas se não confirmar seus dados.",

#     "Titanfall® 2, da sua lista de desejos, está em oferta!",
# ]

# # Testar cada e-mail
# for email in test_emails:
#     inputs = tokenizer(email, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits

#     probs = torch.softmax(logits, dim=-1)
#     predicted_class = torch.argmax(probs, dim=-1).item()  # Usar torch.argmax no tensor PyTorch
#     confidence = probs[0][predicted_class].item() * 100  # Confiança na previsão em porcentagem

#     color = GREEN if predicted_class == 0 else RED  # Verde para legítimo, Vermelho para SPAM

#     print(f"Email: \"{email}\"")
#     print(f"Predicted class: {color + 'SPAM' if predicted_class == 1 else color + 'Legítimo'}")
#     print(f"{RESET}Confidence: {confidence:.2f}%")
#     print(f"Probabilities: {probs}\n{RESET}")


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Códigos ANSI para cores
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

# Carregar o modelo e o tokenizer salvos
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("./models")

# Mover o modelo para a GPU (se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Colocar o modelo em modo de avaliação

# E-mails de teste com assunto
test_emails = [
    {"subject": "Reunião agendada", "content": "Reunião de equipe agendada para amanhã às 10h. Favor confirmar presença."},
    {"subject": "Prêmio ganhado", "content": "Parabéns! Você ganhou um prêmio de R$ 1.000,00. Clique aqui para resgatar."},
    {"subject": "Fatura disponível", "content": "Fatura de cartão de crédito disponível para visualização. Acesse sua conta para mais detalhes."},
    {"subject": "Conta bloqueada", "content": "Sua conta foi bloqueada. Para reativar, clique no link e atualize suas informações."},
    {"subject": "Convite para seminário", "content": "Convite para o seminário sobre segurança cibernética na próxima semana."},
    {"subject": "Oferta exclusiva", "content": "Você foi selecionado para uma oferta exclusiva! Responda agora para garantir."},
    {"subject": "Atualização de software", "content": "Atualização do software disponível. Clique para baixar a versão mais recente."},
    {"subject": "Atenção com a conta", "content": "Atenção! Sua conta será desativada em 24 horas se não confirmar seus dados."},
    {"subject": "Titanfall® 2, da sua lista de desejos, está em oferta!", "content": "1 JOGO DA SUA LISTA DE DESEJOS ESTÁ EM OFERTA! Preços e descontos específicos podem estar sujeitos a alterações. Verifique a página na Loja Steam para mais detalhes. Você está recebendo este e-mail por ter adicionado o item acima à sua lista de desejos no Steam."},
    {"subject": "Você tem uma nova mensagem", "content": "InMail: você tem uma mensagem nova. Clique para ver."},
    {"subject": "Confirmação de pagamento", "content": "Confirmação de pagamento recebida. Obrigado por sua compra."},
    {"subject": "Pix recebido", "content": "Você recebeu um Pix de R$ 50,00 de João Silva. Clique aqui para ver os detalhes."},
    {"subject": "Tigrinho vai te pagar", "content": "Tigrinho vai te pagar R$ 100,00. Clique aqui para ver os detalhes."},
    {"subject": "Bem vindo a nossa plataforma", "content": "Parabéns, você acaba de liberar seu próprio pombo-correio digital! 🐦✉️ Ele está pronto para voar alto e entregar seus e-mails com a eficiência de um... pombo ninja! (Sim, eles existem... pelo menos por aqui 😉) Fique tranquilo, aqui seus e-mails chegam rapidinho (e sem sujeira de pombo nas janelas, prometemos). Agora é só relaxar e deixar que a gente cuide do resto. Se precisar de algo, é só dar um pitiu que o pombo responde! Abraços (e umas asinhas), Equipe do Pombas"}

]

# Testar cada e-mail
for email in test_emails:
    subject = email["subject"]
    content = email["content"]

    # Combinar o assunto e o conteúdo do e-mail
    combined_text = f"Subject: {subject}\n\nContent: {content}"

    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()  # Usar torch.argmax no tensor PyTorch
    confidence = probs[0][predicted_class].item() * 100  # Confiança na previsão em porcentagem

    # Definir cores baseadas na previsão
    color = GREEN if predicted_class == 0 else RED  # Verde para legítimo, Vermelho para SPAM

    # Exibir resultado
    print(f"Subject: \"{subject}\"")
    print(f"Content: \"{content}\"")
    print(f"Predicted class: {color + ('Legítimo' if predicted_class == 0 else 'SPAM') + RESET}")
    print(f"Confidence: {confidence:.2f}%")

    # Exibir probabilidades com formatação
    for i, prob in enumerate(probs[0]):
        label = "Legítimo" if i == 0 else "SPAM"
        print(f"{label} Probability: {prob.item() * 100:.2f}%")

    print(f"{RESET}")

