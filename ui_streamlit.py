import streamlit as st
import pandas as pd
from predictor import predecir_y_comparar_modelos


def main():
    st.set_page_config(page_title="Xibalbá Business", layout="wide")
    st.title("📈 Pronóstico Inteligente de Ventas")

    with st.expander("📝 Instrucciones de uso"):
        st.markdown("""
        Bienvenido a la aplicación de **Predicción Inteligente de Ventas**.

        ### 🧭 Flujo de trabajo:
        1. **Cargar histórico** (Paso 1).
        2. **Presionar el botón verde "Analizar y Sugerir Modelo"** para que la aplicación determine el modelo más confiable  basándose en la validación interna (el **MAE** del último año histórico).
        3. Revisar el **Modelo Recomendado** (Paso 3).
        4. Seleccionar el modelo en el menú desplegable y presionar **"Calcular Pronóstico"** (Paso 4).
        """)

    # ventas_reales_2024 se mantiene como None
    df_historico = None

    # Inicialización de st.session_state
    if 'metricas' not in st.session_state:
        st.session_state.metricas = {}
    if 'df_historico_cargado' not in st.session_state:
        st.session_state.df_historico_cargado = False

    st.sidebar.header("Opciones de Modelo")

    # Selección del modelo para el pronóstico final (Paso 4)
    modelo_elegido = st.sidebar.selectbox(
        "🧠 Selecciona el modelo de predicción:",
        ["Random Forest", "Linear Regression", "XGBoost", "Prophet", "SARIMA"]
    )

    # --- PASO 1: CARGAR HISTÓRICO ---
    archivo_historico = st.sidebar.file_uploader("📂 1. Cargar archivo histórico de ventas", type=["xlsx", "xls"])

    if archivo_historico:
        try:
            df_historico = pd.read_excel(archivo_historico)

            # Normalización de nombres de columnas
            df_historico.columns = [
                col.lower().replace(' ', '_').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó',
                                                                                                            'o').replace(
                    'ú', 'u') for col in df_historico.columns]
            if 'artículo' in df_historico.columns:
                df_historico = df_historico.rename(columns={'artículo': 'articulo'})

            columnas_mensuales = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                                  "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
            columnas_mensuales = [col for col in columnas_mensuales if col in df_historico.columns]
            df_historico["Total Anual"] = df_historico[columnas_mensuales].sum(axis=1)

            st.subheader("📂 Datos históricos cargados con Total Anual")
            columnas_numericas = df_historico.select_dtypes(include=["int", "float"]).columns
            formato_columnas = {col: "{:.0f}" for col in columnas_numericas}
            st.dataframe(df_historico.style.format(formato_columnas))

            # Lógica para refrescar y limpiar las métricas al cargar un nuevo archivo
            if not st.session_state.df_historico_cargado:
                st.session_state.df_historico_cargado = True
                st.session_state.metricas = {}
                # --- CORRECCIÓN: Usar st.rerun() en lugar de st.experimental_rerun() ---
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error al cargar el archivo histórico: {e}")
            df_historico = None
            st.session_state.df_historico_cargado = False

    # --- PASO 2: ANALIZAR Y SUGERIR MODELO (BOTÓN PRINCIPAL) ---
    st.markdown("---")
    if df_historico is not None:

        if st.button("🚀 2. Analizar y Sugerir Modelo Más Confiable"):
            with st.spinner('Analizando y ejecutando validación cruzada...'):
                try:
                    # Se llama a la función con 'None' para el parámetro df_reales
                    _, _, _, _, _, metricas = predecir_y_comparar_modelos(df_historico, None)
                    st.session_state.metricas = metricas

                    if metricas:
                        st.success("✅ Análisis de Modelos Completado. Vaya al Paso 3 para ver las métricas.")

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Se produjo un error durante la comparación de modelos: {e}")

        # --- PASO 3: VISUALIZACIÓN DE MÉTRICAS Y SUGERENCIA ---
        metricas = st.session_state.metricas

        if metricas:
            st.subheader("🔍 3. Resultados de la Validación Histórica")

            # Obtener el año de validación
            año_validacion = metricas[list(metricas.keys())[0]].get('año_validacion', 'último año histórico')
            st.markdown(
                f"**Métricas calculadas al pronosticar el año {año_validacion}** (Modelo entrenado con datos anteriores).")

            for articulo in metricas:
                m = metricas[articulo]
                mae_sarima_display = m.get('mae_sarima', 'N/A')

                st.markdown(f"---")
                st.markdown(f"### Artículo: **{articulo}**")

                # Tabla de MAE para comparación
                mae_data = {
                    'Modelo': ['Linear Regression', 'Random Forest', 'XGBoost', 'Prophet', 'SARIMA'],
                    f'MAE vs. {año_validacion}': [m['mae_lr'], m['mae_rf'], m['mae_xgb'], m['mae_prophet'],
                                                  mae_sarima_display]
                }
                df_mae = pd.DataFrame(mae_data).set_index('Modelo')
                st.dataframe(df_mae)

                # Destacamos la recomendación
                st.markdown(
                    f"## ⭐ Modelo Recomendado para {df_historico['año'].max() + 1}: **{m['modelo_recomendado']}**")
                st.info(
                    f"Ahora, seleccione **{m['modelo_recomendado']}** en el menú desplegable (en la barra lateral) y presione '4. Calcular Pronóstico'.")

    # --- PASO 4: CALCULAR PRONÓSTICO (BOTÓN LATERAL) ---
    st.sidebar.markdown("---")
    if st.sidebar.button("⚙️ 4. Calcular Pronóstico"):
        if df_historico is None:
            st.warning("⚠️ Primero debes cargar los datos históricos.")
        else:
            try:
                # Re-ejecutamos la predicción (se pasa None para df_reales)
                resultados_lr, resultados_rf, resultados_xgb, resultados_prophet, resultados_sarima, metricas = predecir_y_comparar_modelos(
                    df_historico, None)

                # Selección del modelo elegido
                if modelo_elegido == "Random Forest":
                    resultados = resultados_rf
                    mae_key = 'mae_rf'
                elif modelo_elegido == "Linear Regression":
                    resultados = resultados_lr
                    mae_key = 'mae_lr'
                elif modelo_elegido == "XGBoost":
                    resultados = resultados_xgb
                    mae_key = 'mae_xgb'
                elif modelo_elegido == "Prophet":
                    resultados = resultados_prophet
                    mae_key = 'mae_prophet'
                elif modelo_elegido == "SARIMA":
                    resultados = resultados_sarima
                    mae_key = 'mae_sarima'
                else:
                    st.error("❌ Modelo de predicción no válido.")
                    return

                st.success(f"✅ Pronóstico generado usando: {modelo_elegido}")
                año_a_predecir = df_historico['año'].max() + 1

                for articulo, pred in resultados.items():
                    st.write(f"---")
                    st.write(f"### Artículo: {articulo}")

                    pred_df = pd.DataFrame([pred])
                    pred_df.index = [f"Pronóstico {año_a_predecir}"]

                    columnas_mensuales = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                                          "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

                    pred_df = pred_df[columnas_mensuales]
                    pred_df["Total Anual"] = pred_df[columnas_mensuales].sum(axis=1)

                    st.write("#### Predicción por Mes (columnas):")
                    st.dataframe(pred_df.style.format("{:.0f}"))

                    if articulo in metricas:
                        mae_val = metricas[articulo].get(mae_key, 'N/A')
                        año_validacion = metricas[articulo].get('año_validacion', 'último año histórico')
                        st.write(
                            f"**MAE (Error Absoluto Medio) del modelo '{modelo_elegido}' al validar {año_validacion}:** `{mae_val}`"
                        )
                        st.write(f"**Modelo Sugerido:** `{metricas[articulo].get('modelo_recomendado', 'N/A')}`")


            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Se produjo un error al calcular el pronóstico: {e}")


if __name__ == "__main__":
    main()