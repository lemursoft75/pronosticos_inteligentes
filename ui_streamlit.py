import streamlit as st
import pandas as pd
from predictor import predecir_y_comparar_modelos

def main():

    st.set_page_config(page_title="Xibalbá Business", layout="wide")
    st.title("📈 Pronóstico Inteligente de Ventas")


    with st.expander("📝 Instrucciones de uso"):
        st.markdown("""
        Bienvenido a la aplicación de **Predicción Inteligente de Ventas** de Reyma del Sureste.

        ### 🧭 Recomendación general:
        Antes de generar un pronóstico individual, te recomendamos hacer una **Comparativa de Modelos** presionando el botón 🤖 en el menú lateral. Esto te permitirá identificar qué modelo es más confiable para cada artículo con base en el error de predicción (**MAE**).

        ### 🛠️ Pasos para utilizar esta app:
        1. **Cargar archivo histórico**: Sube un archivo Excel con el historial de ventas.
        2. *El archivo debe tener las columnas: artículo, año, y los 12 meses del año.*
        3. *(Opcional)* **Cargar ventas reales** para validar el rendimiento de los modelos.
        4. Presiona el botón **🤖 Comparar Modelos** para ver cuál se ajusta mejor a tus datos.
        5. Luego, selecciona tu modelo favorito en el menú desplegable y presiona **⚙️ Calcular Pronóstico**.
        6. Revisa las predicciones y métricas por artículo.

        ¡Así puedes tomar decisiones más inteligentes y con respaldo estadístico! 📊✨
        """)

    ventas_reales_2024 = None
    df_historico = None
    resultados_globales = {}
    errores_globales = {}
    metricas_globales = {}

    st.sidebar.header("Opciones")

    modelo_elegido = st.sidebar.selectbox(
        "🧠 Selecciona el modelo de predicción:",
        ["Random Forest", "Linear Regression", "XGBoost", "Prophet"]
    )

    archivo_historico = st.sidebar.file_uploader("📂 Cargar archivo histórico de ventas", type=["xlsx", "xls"])
    if archivo_historico:
        df_historico = pd.read_excel(archivo_historico)

        columnas_mensuales = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                              "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        columnas_mensuales = [col for col in columnas_mensuales if col in df_historico.columns]

        # Calcular total anual
        df_historico["Total Anual"] = df_historico[columnas_mensuales].sum(axis=1)

        st.subheader("📂 Datos históricos cargados con Total Anual")

        # Aplicar formato solo a columnas numéricas
        columnas_numericas = df_historico.select_dtypes(include=["int", "float"]).columns
        formato_columnas = {col: "{:.0f}" for col in columnas_numericas}

        st.dataframe(df_historico.style.format(formato_columnas))

    archivo_reales = st.sidebar.file_uploader("📄 (Opcional) Cargar ventas reales", type=["xlsx", "xls"])
    if archivo_reales:
        ventas_reales_2024 = pd.read_excel(archivo_reales)
        st.success("✅ Ventas reales cargadas correctamente.")

        columnas_mensuales_reales = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                                     "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        columnas_mensuales_reales = [col for col in columnas_mensuales_reales if col in ventas_reales_2024.columns]

        ventas_reales_2024["Total Anual"] = ventas_reales_2024[columnas_mensuales_reales].sum(axis=1)

        st.subheader("📄 Ventas reales cargadas con Total Anual")

        columnas_numericas_reales = ventas_reales_2024.select_dtypes(include=["int", "float"]).columns
        formato_reales = {col: "{:.0f}" for col in columnas_numericas_reales}

        st.dataframe(ventas_reales_2024.style.format(formato_reales))

    if st.sidebar.button("🤖 Comparar Modelos"):
        if df_historico is None:
            st.warning("⚠️ Primero debes cargar los datos históricos.")
        else:
            resultados_lr, resultados_rf, resultados_xgb, resultados_prophet, metricas = predecir_y_comparar_modelos(df_historico, ventas_reales_2024)
            st.subheader("🔍 Comparativa entre Modelos")

            for articulo in metricas:
                m = metricas[articulo]
                st.markdown(f"**Artículo: {articulo}**")
                st.markdown(f"- MAE Linear Regression: `{m['mae_lr']}`")
                st.markdown(f"- MAE Random Forest: `{m['mae_rf']}`")
                st.markdown(f"- MAE XGBoost: `{m['mae_xgb']}`")
                st.markdown(f"- MAE Prophet: `{m['mae_prophet']}`")
                st.markdown(f"- ✅ Modelo Recomendado: `{m['modelo_recomendado']}`")

    if st.sidebar.button("⚙️ Calcular Pronóstico"):
        if df_historico is None:
            st.warning("⚠️ Primero debes cargar los datos históricos.")
        else:
            resultados_lr, resultados_rf, resultados_xgb, resultados_prophet, metricas = predecir_y_comparar_modelos(df_historico, ventas_reales_2024)

            if modelo_elegido == "Random Forest":
                resultados = resultados_rf
            elif modelo_elegido == "Linear Regression":
                resultados = resultados_lr
            elif modelo_elegido == "XGBoost":
                resultados = resultados_xgb
            elif modelo_elegido == "Prophet":
                resultados = resultados_prophet
            else:
                st.error("❌ Modelo de predicción no válido.")
                resultados = {}

            resultados_globales.update(resultados)

            for articulo in resultados.keys():
                if articulo in metricas:
                    errores_globales[articulo] = (
                        metricas[articulo]['mae_rf'] if modelo_elegido == "Random Forest"
                        else metricas[articulo]['mae_lr'] if modelo_elegido == "Linear Regression"
                        else metricas[articulo]['mae_xgb'] if modelo_elegido == "XGBoost"
                        else metricas[articulo]['mae_prophet']
                    )

            metricas_globales.update({
                a: {
                    **metricas.get(a, {}),
                    "modelo_recomendado": modelo_elegido
                } for a in resultados
            })

            st.success(f"✅ Pronóstico generado usando: {modelo_elegido}")

            for articulo, pred in resultados.items():
                st.write(f"### Artículo: {articulo}")
                pred_df = pd.DataFrame(pred, index=[0])
                pred_df["Total Anual"] = pred_df.sum(axis=1)
                st.write("#### Predicción por Mes (columnas):")
                st.dataframe(pred_df.style.format("{:.0f}"))

                if articulo in errores_globales:
                    st.write(f"MAE: {errores_globales[articulo]}")

if __name__ == "__main__":
    main()