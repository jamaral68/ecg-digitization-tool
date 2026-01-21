import boto3
from io import BytesIO
from PIL import Image

# Defina o nome do bucket e o prefixo (diretório) no S3
bucket_name = 'nome-do-seu-bucket'
prefix = 'caminho/para/as/imagens/'

# Crie o cliente S3
s3 = boto3.client('s3')

# Liste os objetos no bucket dentro do prefixo
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Verifique se há objetos no bucket
if 'Contents' in response:
    for obj in response['Contents']:
        file_key = obj['Key']
        print(f'Processando: {file_key}')
        
        # Baixe o arquivo do S3
        s3_object = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = s3_object['Body'].read()
        
        # Abra a imagem usando o Pillow
        image = Image.open(BytesIO(file_content))
        image.show()  # Mostra a imagem
else:
    print(f'Nenhum objeto encontrado no bucket {bucket_name} com o prefixo {prefix}')
