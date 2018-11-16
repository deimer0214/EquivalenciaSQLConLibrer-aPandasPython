
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
archivo_base = pd.read_csv("knn_file.csv")    #ReadTheData
archivo_train = pd.read_csv("knn_file_1.csv")    #ReadTheData


# In[46]:


print(archivo_base)
print(archivo_train)


# In[7]:


archivo_base["maximo"]=archivo_base[["a1","a2","a3","a4","a5","a6"]].max(axis=1)
archivo_base["minimo"]=archivo_base[["a1","a2","a3","a4","a5","a6"]].min(axis=1)
print(archivo_base)
archivo_train["maximo"]=archivo_train[["a1","a2","a3","a4","a5","a6"]].max(axis=1)
archivo_train["minimo"]=archivo_train[["a1","a2","a3","a4","a5","a6"]].min(axis=1)
print(archivo_train)


# In[8]:


archivo_base["b1"]=(archivo_base["a1"]-archivo_base["minimo"])/(archivo_base["maximo"]-archivo_base["minimo"])
archivo_base["b2"]=(archivo_base["a2"]-archivo_base["minimo"])/(archivo_base["maximo"]-archivo_base["minimo"])
archivo_base["b3"]=(archivo_base["a3"]-archivo_base["minimo"])/(archivo_base["maximo"]-archivo_base["minimo"])
archivo_base["b4"]=(archivo_base["a4"]-archivo_base["minimo"])/(archivo_base["maximo"]-archivo_base["minimo"])
archivo_base["b5"]=(archivo_base["a5"]-archivo_base["minimo"])/(archivo_base["maximo"]-archivo_base["minimo"])
archivo_base["b6"]=(archivo_base["a6"]-archivo_base["minimo"])/(archivo_base["maximo"]-archivo_base["minimo"])
print(archivo_base)
archivo_train["b1"]=(archivo_train["a1"]-archivo_train["minimo"])/(archivo_train["maximo"]-archivo_train["minimo"])
archivo_train["b2"]=(archivo_train["a2"]-archivo_train["minimo"])/(archivo_train["maximo"]-archivo_train["minimo"])
archivo_train["b3"]=(archivo_train["a3"]-archivo_train["minimo"])/(archivo_train["maximo"]-archivo_train["minimo"])
archivo_train["b4"]=(archivo_train["a4"]-archivo_train["minimo"])/(archivo_train["maximo"]-archivo_train["minimo"])
archivo_train["b5"]=(archivo_train["a5"]-archivo_train["minimo"])/(archivo_train["maximo"]-archivo_train["minimo"])
archivo_train["b6"]=(archivo_train["a6"]-archivo_train["minimo"])/(archivo_train["maximo"]-archivo_train["minimo"])
print(archivo_train)


# In[9]:


normalizado_base=archivo_base[['PK_1','Invertir','b1','b2','b3','b4','b5','b6']]
print(normalizado_base)
normalizado_train=archivo_train[['PK_2','b1','b2','b3','b4','b5','b6']]
print(normalizado_train)


# In[10]:


normalizado_base['join']=1
print(normalizado_base)
normalizado_train['join']=1
print(normalizado_train)


# In[11]:


full_datos = pd.merge(normalizado_base, normalizado_train, on=['join'])
print(full_datos)


# In[21]:


metrica2=full_datos
metrica2['tipo_metrica']='euclidean'
metrica2['valor_metrica_1']=(metrica2["b1_x"]-metrica2["b1_y"])**2
metrica2['valor_metrica_2']=(metrica2["b2_x"]-metrica2["b2_y"])**2
metrica2['valor_metrica_3']=(metrica2["b3_x"]-metrica2["b3_y"])**2
metrica2['valor_metrica_4']=(metrica2["b4_x"]-metrica2["b4_y"])**2
metrica2['valor_metrica_5']=(metrica2["b5_x"]-metrica2["b5_y"])**2
metrica2['valor_metrica_6']=(metrica2["b6_x"]-metrica2["b6_y"])**2
#inicio_SumaTotal
metrica2['valor_metrica']=np.sqrt(metrica2['valor_metrica_1']
+metrica2['valor_metrica_2']
+metrica2['valor_metrica_3']
+metrica2['valor_metrica_4']
+metrica2['valor_metrica_5']
+metrica2['valor_metrica_6'])
metrica2=full_datos[['tipo_metrica','PK_2','valor_metrica','PK_1']]
metrica2=metrica2.sort_values(by=['tipo_metrica','PK_2','valor_metrica'])
print(metrica2)


# In[28]:


metrica2_PK_01=metrica2[metrica2['PK_2']=='M01']
metrica2_PK_01=metrica2_PK_01.reset_index(drop=True)
metrica2_PK_01['numero']=metrica2_PK_01.index+1
print(metrica2_PK_01[metrica2_PK_01['numero']==1])
print('Para K=1')
temp1=metrica2_PK_01[metrica2_PK_01['numero']==1]
temp2=metrica2_PK_01[metrica2_PK_01['numero']==2]
temp3=metrica2_PK_01[metrica2_PK_01['numero']==3]
temp4=metrica2_PK_01[metrica2_PK_01['numero']==4]
temp5=metrica2_PK_01[metrica2_PK_01['numero']==5]
rename=metrica2_PK_01.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[29]:


metrica2_PK_02=metrica2[metrica2['PK_2']=='M02']
metrica2_PK_02=metrica2_PK_02.reset_index(drop=True)
metrica2_PK_02['numero']=metrica2_PK_02.index+1
print(metrica2_PK_02[metrica2_PK_02['numero']==1])
print('Para K=1')
temp1=metrica2_PK_02[metrica2_PK_02['numero']==1]
temp2=metrica2_PK_02[metrica2_PK_02['numero']==2]
temp3=metrica2_PK_02[metrica2_PK_02['numero']==3]
temp4=metrica2_PK_02[metrica2_PK_02['numero']==4]
temp5=metrica2_PK_02[metrica2_PK_02['numero']==5]
rename=metrica2_PK_02.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[31]:


metrica2_PK_03=metrica2[metrica2['PK_2']=='M03']
metrica2_PK_03=metrica2_PK_03.reset_index(drop=True)
metrica2_PK_03['numero']=metrica2_PK_03.index+1
print(metrica2_PK_03[metrica2_PK_03['numero']==1])
print('Para K=1')
temp1=metrica2_PK_03[metrica2_PK_03['numero']==1]
temp2=metrica2_PK_03[metrica2_PK_03['numero']==2]
temp3=metrica2_PK_03[metrica2_PK_03['numero']==3]
temp4=metrica2_PK_03[metrica2_PK_03['numero']==4]
temp5=metrica2_PK_03[metrica2_PK_03['numero']==5]
rename=metrica2_PK_03.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[32]:


metrica2_PK_04=metrica2[metrica2['PK_2']=='M04']
metrica2_PK_04=metrica2_PK_04.reset_index(drop=True)
metrica2_PK_04['numero']=metrica2_PK_04.index+1
print(metrica2_PK_04[metrica2_PK_04['numero']==1])
print('Para K=1')
temp1=metrica2_PK_04[metrica2_PK_04['numero']==1]
temp2=metrica2_PK_04[metrica2_PK_04['numero']==2]
temp3=metrica2_PK_04[metrica2_PK_04['numero']==3]
temp4=metrica2_PK_04[metrica2_PK_04['numero']==4]
temp5=metrica2_PK_04[metrica2_PK_04['numero']==5]
rename=metrica2_PK_04.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[25]:


metricainf=full_datos
metricainf['tipo_metrica']='distinf'
metricainf["tmp1"]=abs(metricainf["b1_x"]-metricainf["b1_y"])
metricainf["tmp2"]=abs(metricainf["b2_x"]-metricainf["b2_y"])
metricainf["tmp3"]=abs(metricainf["b3_x"]-metricainf["b3_y"])
metricainf["tmp4"]=abs(metricainf["b4_x"]-metricainf["b4_y"])
metricainf["tmp5"]=abs(metricainf["b5_x"]-metricainf["b5_y"])
metricainf["tmp6"]=abs(metricainf["b6_x"]-metricainf["b6_y"])
#inicio_SumaTotal
metricainf["distinf"] = metricainf[["tmp1","tmp2","tmp3","tmp4","tmp5","tmp6"]].max(axis=1)
metricainf=metricainf[['tipo_metrica','PK_2','valor_metrica','PK_1']]
metricainf=metricainf.sort_values(by=['tipo_metrica','PK_2','valor_metrica'])
print(metricainf)


# In[33]:


metricainf_PK_01=metricainf[metricainf['PK_2']=='M01']
metricainf_PK_01=metricainf_PK_01.reset_index(drop=True)
metricainf_PK_01['numero']=metricainf_PK_01.index+1
print(metricainf_PK_01[metricainf_PK_01['numero']==1])
print('Para K=1')
temp1=metricainf_PK_01[metricainf_PK_01['numero']==1]
temp2=metricainf_PK_01[metricainf_PK_01['numero']==2]
temp3=metricainf_PK_01[metricainf_PK_01['numero']==3]
temp4=metricainf_PK_01[metricainf_PK_01['numero']==4]
temp5=metricainf_PK_01[metricainf_PK_01['numero']==5]
rename=metricainf_PK_01.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[34]:


metricainf_PK_02=metricainf[metricainf['PK_2']=='M02']
metricainf_PK_02=metricainf_PK_02.reset_index(drop=True)
metricainf_PK_02['numero']=metricainf_PK_02.index+1
print(metricainf_PK_02[metricainf_PK_02['numero']==1])
print('Para K=1')
temp1=metricainf_PK_02[metricainf_PK_02['numero']==1]
temp2=metricainf_PK_02[metricainf_PK_02['numero']==2]
temp3=metricainf_PK_02[metricainf_PK_02['numero']==3]
temp4=metricainf_PK_02[metricainf_PK_02['numero']==4]
temp5=metricainf_PK_02[metricainf_PK_02['numero']==5]
rename=metricainf_PK_02.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[471]:


metricainf_PK_03=metricainf[metricainf['PK_2']=='M03']
metricainf_PK_03=metricainf_PK_03.reset_index(drop=True)
metricainf_PK_03['numero']=metricainf_PK_03.index+1
print(metricainf_PK_03[metricainf_PK_03['numero']==1])
print('Para K=1')
temp1=metricainf_PK_03[metricainf_PK_03['numero']==1]
temp2=metricainf_PK_03[metricainf_PK_03['numero']==2]
temp3=metricainf_PK_03[metricainf_PK_03['numero']==3]
temp4=metricainf_PK_03[metricainf_PK_03['numero']==4]
temp5=metricainf_PK_03[metricainf_PK_03['numero']==5]
rename=archivo_salida.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[37]:


metricainf_PK_04=metricainf[metricainf['PK_2']=='M04']
metricainf_PK_04=metricainf_PK_04.reset_index(drop=True)
metricainf_PK_04['numero']=metricainf_PK_04.index+1
print(metricainf_PK_04[metricainf_PK_04['numero']==1])
print('Para K=1')
temp1=metricainf_PK_04[metricainf_PK_04['numero']==1]
temp2=metricainf_PK_04[metricainf_PK_04['numero']==2]
temp3=metricainf_PK_04[metricainf_PK_04['numero']==3]
temp4=metricainf_PK_04[metricainf_PK_04['numero']==4]
temp5=metricainf_PK_04[metricainf_PK_04['numero']==5]
rename=metricainf_PK_04.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[38]:


metrica1=full_datos
metrica1['tipo_metrica']='manhatten'
metrica1["tmp1"]=abs(metrica1["b1_x"]-metrica1["b1_y"])
metrica1["tmp2"]=abs(metrica1["b2_x"]-metrica1["b2_y"])
metrica1["tmp3"]=abs(metrica1["b3_x"]-metrica1["b3_y"])
metrica1["tmp4"]=abs(metrica1["b4_x"]-metrica1["b4_y"])
metrica1["tmp5"]=abs(metrica1["b5_x"]-metrica1["b5_y"])
metrica1["tmp6"]=abs(metrica1["b6_x"]-metrica1["b6_y"])
#inicio_SumaTotal
metrica1["valor_metrica"] = metrica1["tmp1"]+metrica1["tmp2"]+metrica1["tmp3"]+metrica1["tmp4"]+metrica1["tmp5"]+metrica1["tmp6"]
metrica1=metrica1[['tipo_metrica','PK_2','valor_metrica','PK_1']]
metrica1=metrica1.sort_values(by=['tipo_metrica','PK_2','valor_metrica'])
print(metrica1)


# In[40]:


metrica1_PK_01=metrica1[metrica1['PK_2']=='M01']
metrica1_PK_01=metrica1_PK_01.reset_index(drop=True)
metrica1_PK_01['numero']=metrica1_PK_01.index+1
print(metrica1_PK_01[metrica1_PK_01['numero']==1])
print('Para K=1')
temp1=metrica1_PK_01[metrica1_PK_01['numero']==1]
temp2=metrica1_PK_01[metrica1_PK_01['numero']==2]
temp3=metrica1_PK_01[metrica1_PK_01['numero']==3]
temp4=metrica1_PK_01[metrica1_PK_01['numero']==4]
temp5=metrica1_PK_01[metrica1_PK_01['numero']==5]
rename=metrica1_PK_01.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[41]:


metrica1_PK_02=metrica1[metrica1['PK_2']=='M02']
metrica1_PK_02=metrica1_PK_02.reset_index(drop=True)
metrica1_PK_02['numero']=metrica1_PK_02.index+1
print(metrica1_PK_02[metrica1_PK_02['numero']==1])
print('Para K=1')
temp1=metrica1_PK_02[metrica1_PK_02['numero']==1]
temp2=metrica1_PK_02[metrica1_PK_02['numero']==2]
temp3=metrica1_PK_02[metrica1_PK_02['numero']==3]
temp4=metrica1_PK_02[metrica1_PK_02['numero']==4]
temp5=metrica1_PK_02[metrica1_PK_02['numero']==5]
rename=metrica1_PK_02.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[45]:


metrica1_PK_03=metrica1[metrica1['PK_2']=='M03']
metrica1_PK_03=metrica1_PK_03.reset_index(drop=True)
metrica1_PK_03['numero']=metrica1_PK_03.index+1

print(metrica1_PK_03[metrica1_PK_03['numero']==1])
print('Para K=1')
temp1=metrica1_PK_03[metrica1_PK_03['numero']==1]
temp2=metrica1_PK_03[metrica1_PK_03['numero']==2]
temp3=metrica1_PK_03[metrica1_PK_03['numero']==3]
temp4=metrica1_PK_03[metrica1_PK_03['numero']==4]
temp5=metrica1_PK_03[metrica1_PK_03['numero']==5]
rename=metrica1_PK_03.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]
print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)
resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]
print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[93]:


metrica1_PK_04=metrica1[metrica1['PK_2']=='M04']
metrica1_PK_04=metrica1_PK_04.reset_index(drop=True)
metrica1_PK_04['numero']=metrica1_PK_04.index+1

print(metrica1_PK_04[metrica1_PK_04['numero']==1])
print('Para K=1')
temp1=metrica1_PK_04[metrica1_PK_04['numero']==1]
temp2=metrica1_PK_04[metrica1_PK_04['numero']==2]
temp3=metrica1_PK_04[metrica1_PK_04['numero']==3]
temp4=metrica1_PK_04[metrica1_PK_04['numero']==4]
temp5=metrica1_PK_04[metrica1_PK_04['numero']==5]
rename=metrica1_PK_04.rename(index=str, columns={"a1": "Precio", "a2": "Metros2","a3": "Baños","a4": "Cuartos","a5": "Estacionamiento","a6": "Mantenimiento"})
resul=temp1['PK_1'].to_string(index=False)
temp_b=rename[rename['PK_1']==resul]
print(temp_b)
resul=temp2['PK_1'].to_string(index=False)
temp_b2=rename[rename['PK_1']==resul]
resul=temp3['PK_1'].to_string(index=False)
temp_b3=rename[rename['PK_1']==resul]

print('Para K=3')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3])
print(tmp_final)

resul=temp4['PK_1'].to_string(index=False)
temp_b4=rename[rename['PK_1']==resul]
resul=temp5['PK_1'].to_string(index=False)
temp_b5=rename[rename['PK_1']==resul]

print('Para K=5')
tmp_final=pd.concat([temp_b,temp_b2,temp_b3,temp_b4,temp_b5])
print(tmp_final)


# In[149]:


archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metrica1_PK_04_1=metrica1_PK_04[['PK_1','numero','PK_2','valor_metrica']]
metrica1_PK_04_f3 = pd.merge(archivo_base_1, metrica1_PK_04_1, on=['PK_1'])
metrica1_PK_04_f3=metrica1_PK_04_f3.sort_values(by=['valor_metrica'])
metrica1_PK_04_f3=metrica1_PK_04_f3.head(1)
metrica1_PK_04_f3=metrica1_PK_04_f3.replace(['Si', 'No'],[1,0])
print(metrica1_PK_04_f3)
a=metrica1_PK_04_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metrica1_PK_04_f3[['PK_1']][metrica1_PK_04_f3['Invertir']==1])
print('el porcentaje de clasificación es',a)


# In[151]:


archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metrica1_PK_04_1=metrica1_PK_04[['PK_1','numero','PK_2','valor_metrica']]
metrica1_PK_04_f3 = pd.merge(archivo_base_1, metrica1_PK_04_1, on=['PK_1'])
metrica1_PK_04_f3=metrica1_PK_04_f3.sort_values(by=['valor_metrica'])
metrica1_PK_04_f3=metrica1_PK_04_f3.head(3)
metrica1_PK_04_f3=metrica1_PK_04_f3.replace(['Si', 'No'],[1,0])
print(metrica1_PK_04_f3)
a=metrica1_PK_04_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metrica1_PK_04_f3[['PK_1']][metrica1_PK_04_f3['Invertir']==1])
print('el porcentaje de clasificación es',a/3)


# In[150]:


archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metrica1_PK_04_1=metrica1_PK_04[['PK_1','numero','PK_2','valor_metrica']]
metrica1_PK_04_f3 = pd.merge(archivo_base_1, metrica1_PK_04_1, on=['PK_1'])
metrica1_PK_04_f3=metrica1_PK_04_f3.sort_values(by=['valor_metrica'])
metrica1_PK_04_f3=metrica1_PK_04_f3.head(5)
metrica1_PK_04_f3=metrica1_PK_04_f3.replace(['Si', 'No'],[1,0])
print(metrica1_PK_04_f3)
a=metrica1_PK_04_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metrica1_PK_04_f3[['PK_1']][metrica1_PK_04_f3['Invertir']==1])
print('el porcentaje de clasificación es',a/5)


# In[153]:


#M03 con K=1
archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metrica1_PK_03_1=metrica1_PK_03[['PK_1','numero','PK_2','valor_metrica']]
metrica1_PK_03_f3 = pd.merge(archivo_base_1, metrica1_PK_03_1, on=['PK_1'])
metrica1_PK_03_f3=metrica1_PK_03_f3.sort_values(by=['valor_metrica'])
metrica1_PK_03_f3=metrica1_PK_03_f3.head(1)
metrica1_PK_03_f3=metrica1_PK_03_f3.replace(['Si', 'No'],[1,0])
print(metrica1_PK_03_f3)
a=metrica1_PK_03_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metrica1_PK_03_f3[['PK_1']][metrica1_PK_03_f3['Invertir']==1])
print('el porcentaje de clasificación es',a)


# In[155]:


#M03 con K=3
archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metrica1_PK_03_1=metrica1_PK_03[['PK_1','numero','PK_2','valor_metrica']]
metrica1_PK_03_f3 = pd.merge(archivo_base_1, metrica1_PK_03_1, on=['PK_1'])
metrica1_PK_03_f3=metrica1_PK_03_f3.sort_values(by=['valor_metrica'])
metrica1_PK_03_f3=metrica1_PK_03_f3.head(3)
metrica1_PK_03_f3=metrica1_PK_03_f3.replace(['Si', 'No'],[1,0])
print(metrica1_PK_03_f3)
a=metrica1_PK_03_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metrica1_PK_03_f3[['PK_1']][metrica1_PK_03_f3['Invertir']==1])
print('el porcentaje de clasificación es',a/3)


# In[156]:


#M03 con K=5
archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metrica1_PK_03_1=metrica1_PK_03[['PK_1','numero','PK_2','valor_metrica']]
metrica1_PK_03_f3 = pd.merge(archivo_base_1, metrica1_PK_03_1, on=['PK_1'])
metrica1_PK_03_f3=metrica1_PK_03_f3.sort_values(by=['valor_metrica'])
metrica1_PK_03_f3=metrica1_PK_03_f3.head(5)
metrica1_PK_03_f3=metrica1_PK_03_f3.replace(['Si', 'No'],[1,0])
print(metrica1_PK_03_f3)
a=metrica1_PK_03_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metrica1_PK_03_f3[['PK_1']][metrica1_PK_03_f3['Invertir']==1])
print('el porcentaje de clasificación es',a/5)


# In[165]:


#M02 con K=1
archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metrica1_PK_02_1=metrica1_PK_02[['PK_1','numero','PK_2','valor_metrica']]
metrica1_PK_02_f3 = pd.merge(archivo_base_1, metrica1_PK_02_1, on=['PK_1'])
metrica1_PK_02_f3=metrica1_PK_02_f3.sort_values(by=['valor_metrica'])
metrica1_PK_02_f3=metrica1_PK_02_f3.head(1)
metrica1_PK_02_f3=metrica1_PK_02_f3.replace(['Si', 'No'],[1,0])
print(metrica1_PK_02_f3)
a=metrica1_PK_02_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metrica1_PK_02_f3[['PK_1']][metrica1_PK_02_f3['Invertir']==1])
print('el porcentaje de clasificación es',a/1)


# In[166]:


#M02 con K=3
archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metrica1_PK_02_1=metrica1_PK_02[['PK_1','numero','PK_2','valor_metrica']]
metrica1_PK_02_f3 = pd.merge(archivo_base_1, metrica1_PK_02_1, on=['PK_1'])
metrica1_PK_02_f3=metrica1_PK_02_f3.sort_values(by=['valor_metrica'])
metrica1_PK_02_f3=metrica1_PK_02_f3.head(3)
metrica1_PK_02_f3=metrica1_PK_02_f3.replace(['Si', 'No'],[1,0])
print(metrica1_PK_02_f3)
a=metrica1_PK_02_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metrica1_PK_02_f3[['PK_1']][metrica1_PK_02_f3['Invertir']==1])
print('el porcentaje de clasificación es',a/3)


# In[162]:


#M01 con K=5
archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metrica1_PK_02_1=metrica1_PK_02[['PK_1','numero','PK_2','valor_metrica']]
metrica1_PK_02_f3 = pd.merge(archivo_base_1, metrica1_PK_02_1, on=['PK_1'])
metrica1_PK_02_f3=metrica1_PK_02_f3.sort_values(by=['valor_metrica'])
metrica1_PK_02_f3=metrica1_PK_02_f3.head(5)
metrica1_PK_02_f3=metrica1_PK_02_f3.replace(['Si', 'No'],[1,0])
print(metrica1_PK_02_f3)
a=metrica1_PK_02_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metrica1_PK_02_f3[['PK_1']][metrica1_PK_02_f3['Invertir']==1])
print('el porcentaje de clasificación es',a/5)


# In[211]:


#M03 con K=1
archivo_base_1=archivo_base[['PK_1','a1','a2','a3','a4','a5','a6','Invertir']]
metricainf_PK_03_1=metricainf_PK_03[['PK_1','numero','PK_2','valor_metrica']]
metricainf_PK_03_f3= pd.merge(archivo_base_1, metricainf_PK_03_1, on=['PK_1'])
metricainf_PK_02_f3=metricainf_PK_02_f3.sort_values(by=['valor_metrica'])
metricainf_PK_02_f3=metricainf_PK_02_f3.head(1)
metricainf_PK_02_f3=metricainf_PK_02_f3.replace(['Si', 'No'],[1,0])
print(metricainf_PK_02_f3)
a=metricainf_PK_02_f3['Invertir'].sum()
print('Las variables de clasificacion son \n ')
print(metricainf_PK_02_f3[['PK_1']][metricainf_PK_02_f3['Invertir']==1])
print('el porcentaje de clasificación es',a/1)

