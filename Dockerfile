FROM python:3.9-slim

# Définition du répertoire de travail
WORKDIR /app

# Copie des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code
COPY . /app

# On expose les 2 ports nécessaires
EXPOSE 8501
EXPOSE 8000

# On s'assure que start.sh est exécutable
RUN chmod +x start.sh

# Commande de lancement
CMD ["./start.sh"]
