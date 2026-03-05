import os
import csv

# Caminho da pasta onde estão os arquivos PNG
folder_path = '/home/anderson/ecg/ecg-digitization-tool/bucket'

# Filtra apenas arquivos .png (não percorre subpastas)
png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png') and os.path.isfile(os.path.join(folder_path, f))]

# Caminho do arquivo CSV de saída
output_csv = 'lista_pngs.csv'

# Escreve os nomes dos arquivos no CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name'])  # Cabeçalho
    for file_name in png_files:
        writer.writerow([file_name])

print(f'CSV com arquivos .png salvo como: {output_csv}')