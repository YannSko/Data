df = pd.read_csv("./DataSet/Analyse_sale/Sales_January_2019.csv")
files = [file for file in os.listdir("./DataSet/Analyse_sale")if file != '.ipynb_checkpoints']
all_months_data= pd.DataFrame()
for file in files:
    df = pd.read_csv("./DataSet/Analyse_sale/"+file)
    all_monts_data = pd.concat ([all_monts_data, df])
all_months_data.to_csv("./DataSet/Analyse_sale/all_data.csv", index=False)
    
                                     