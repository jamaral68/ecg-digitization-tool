import boto3
import os

# Defina o nome do bucket e o diretório (prefixo) no S3
bucket_name = 'nome-do-seu-bucket'
destination_prefix = 'caminho/para/salvar/as/imagens/'

# Crie o cliente S3
s3 = boto3.client('s3')

# Diretório local no SageMaker Studio onde as imagens estão armazenadas
local_directory = '/caminho/local/das/imagens/'

# Liste os arquivos no diretório local
for file_name in os.listdir(local_directory):
    local_path = os.path.join(local_directory, file_name)
    
    # Verifique se é um arquivo (não uma pasta)
    if os.path.isfile(local_path):
        # Chave do arquivo no bucket S3
        s3_key = destination_prefix + file_name
        print(f'Fazendo upload de {local_path} para s3://{bucket_name}/{s3_key}')
        
        # Upload do arquivo para o S3
        s3.upload_file(local_path, bucket_name, s3_key)

print('Upload concluído!')
