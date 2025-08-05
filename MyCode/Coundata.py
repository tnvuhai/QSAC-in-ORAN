import pandas as pd
import os

lte_total_rows = 0
nr_total_rows = 0

lte_path = './Data/'  # Đường dẫn đến thư mục chứa file LTE_demand_{i}.xlsx
nr_path = './Data/'   # Đường dẫn đến thư mục chứa file NR_demand_{i}.xlsx

# Lặp qua các file từ 1 đến 49
for i in range(1, 50):
    lte_file = os.path.join(lte_path, f'LTE_demand_{i}.xlsx')
    nr_file = os.path.join(nr_path, f'NR_demand_{i}.xlsx')

    # Kiểm tra và load file LTE
    if os.path.exists(lte_file):
        lte_df = pd.read_excel(lte_file)
        rows = len(lte_df)
        lte_total_rows += rows
        print(f'✅ Đã đọc {lte_file}: {rows} dòng')
    else:
        print(f'⚠️ Không tìm thấy {lte_file}')

    # Kiểm tra và load file NR
    if os.path.exists(nr_file):
        nr_df = pd.read_excel(nr_file)
        rows = len(nr_df)
        nr_total_rows += rows
        print(f'✅ Đã đọc {nr_file}: {rows} dòng')
    else:
        print(f'⚠️ Không tìm thấy {nr_file}')

print('\n📊 Tổng kết:')
print(f'Tổng số bản ghi LTE: {lte_total_rows}')
print(f'Tổng số bản ghi NR: {nr_total_rows}')
