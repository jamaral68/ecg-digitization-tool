import boto3

# Defina o nome do bucket e o prefixo (diretório) no S3
bucket_name = 'nome-do-seu-bucket'
prefix = 'caminho/para/as/imagens/'

# Crie o cliente S3
s3 = boto3.client('s3')

# Liste os objetos no bucket dentro do prefixo
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Verifique se há objetos no bucket e liste os nomes
if 'Contents' in response:
    print('Arquivos disponíveis no bucket:')
    file_names = []
    for obj in response['Contents']:
        file_key = obj['Key']
        file_names.append(file_key)
        print(file_key)
    
    # Solicite ao usuário que escolha um arquivo
    chosen_file_name = input('Digite o nome do arquivo que deseja baixar: ')
    
    if chosen_file_name in file_names:
        print(f'Baixando {chosen_file_name}...')
        
        # Baixe o arquivo para o diretório local
        local_file_path = f'./{chosen_file_name.split("/")[-1]}'
        s3.download_file(bucket_name, chosen_file_name, local_file_path)
        print(f'{chosen_file_name} foi salvo em {local_file_path}')
    else:
        print('O arquivo especificado não foi encontrado na lista.')
else:
    print(f'Nenhum objeto encontrado no bucket {bucket_name} com o prefixo {prefix}')
