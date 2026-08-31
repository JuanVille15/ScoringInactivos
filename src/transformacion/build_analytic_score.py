"""Transformación: arma la base analítica de scoring D1-D5 (severidad, enganche,
recencia, vínculo, externo) para la población de inactivos, a partir de las
bases crudas que entrega `extract_raw`.

Salida: columnas ['Periodo', 'Id',
'Tiempo_Inactividad_Meses', 'Factura_Total_Promedio_GECC_6M',
'Recaudo_Total_Promedio_GECC_6M', 'Oferta_Reactivacion_Disponible',
'intenciones_retiro_1y',
'Numcantidadproductos', 'Cantidad_empresas', 'Valor_perseverancia_vs_ingresos',
'Total_Eventos', 'Valor_Capitalizado',
'Distancia_ultima_reactivacion', 'Recencia_ultimo_producto',
'Turnos_En_Oficinas_Total_Ult12Meses', 'Cantidad_pqr_ultimo_anno',
'Cuotas_pagadas_vs_antiguedad', 'Cantidad_Reactivaciones_Previas', 'Clv',
'Saldoaportes', 'Perseverancia_Cerca',
'Alerta_Habito_Pago_Externo', 'Alerta_Estado_Creditos_Externos',
'Alerta_Capacidad_Pago_Externo'].
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _id_a_str(series: pd.Series) -> pd.Series:
    """Normaliza una columna identificadora (cédula) a str, sin sufijo '.0'.

    Las columnas de identificación numéricas que llegan desde SQL Server a
    veces se tipan como float64 (ej. cuando la columna origen admite NULL),
    lo que produce sufijos '.0' al castear directo a str (ej. '7601445.0' en
    vez de '7601445') y rompe silenciosamente cualquier merge por Id, sin
    lanzar ningún error — el merge simplemente no encuentra coincidencias.
    Pasar primero por Int64 (nullable) elimina ese sufijo sin perder los NaN
    reales (quedan como NaN, no como el string 'nan').

    Args:
        series: Serie con identificadores, en cualquier dtype (str, int64,
            float64).

    Returns:
        pd.Series de dtype str (con NaN reales donde el valor no era numérico
        o era nulo).
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def _poblacion_normalizada(poblacion: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la base insumo a columnas ['Periodo', 'Id'] tipo str.

    Args:
        poblacion: DataFrame leído de `Poblacion_Inactivos.xlsx`, con columnas
            ``['Cedula', 'Periodo']``.

    Returns:
        Copia con columnas renombradas a ``['Periodo', 'Id']``, ambas como str.
    """
    return poblacion.rename(columns={"Cedula": "Id"}).assign(
        Id=lambda d: _id_a_str(d["Id"]),
        Periodo=lambda d: d["Periodo"].astype(str),
    )[["Periodo", "Id"]]


def _a_binario(series: pd.Series) -> pd.Series:
    """Convierte una columna binaria de ConsultaIntegral360 (Si/No, 1/0, bool) a bool.

    Args:
        series: Serie con valores en cualquiera de los formatos ``Si/No``,
            ``SI/NO``, ``si/no``, ``1/0`` o ya booleanos.

    Returns:
        Serie booleana. Valores no reconocidos quedan en ``False``.
    """
    if series.dtype == bool:
        return series
    mapa = {
        "No": False, "NO": False, "no": False,
        "Si": True, "SI": True, "si": True,
        "1": True, "0": False, 1: True, 0: False,
    }
    return series.map(mapa).fillna(False).astype(bool)


def calcular_numcantidadproductos(cantidad_productos: pd.DataFrame) -> pd.DataFrame:
    """Extrae el snapshot puntual de tenencia de productos (D2).

    Args:
        cantidad_productos: DataFrame de ``bases['cantidad_productos']`` con
            columnas ``['Periodo', 'Id', 'Numcantidadproductos']`` (Periodo es
            el mes anterior al periodo objetivo, fijado en la extracción).

    Returns:
        DataFrame con columnas ``['Id', 'Numcantidadproductos']``, un registro
        por cédula.
    """
    return (
        cantidad_productos
        .assign(Id=lambda d: _id_a_str(d["Id"]))
        [["Id", "Numcantidadproductos"]]
        .drop_duplicates(subset="Id", keep="first")
    )


def calcular_cantidad_empresas(v_360: pd.DataFrame) -> pd.DataFrame:
    """Calcula la cantidad de empresas con las que el asociado tiene vínculo (D2).

    Suma las 4 columnas booleanas ``Tipo_cliente_*`` de la vista 360.

    Args:
        v_360: DataFrame de ``bases['v_360']``.

    Returns:
        DataFrame con columnas ``['Id', 'Cantidad_empresas']``.
    """
    df = v_360.assign(Id=lambda d: _id_a_str(d["Identificacion"]))

    for col in [
        "Tipo_cliente_bancoomeva", "Tipo_cliente_prepagada",
        "Tipo_cliente_seguros", "Tipo_cliente_adicionales",
    ]:
        df[col] = _a_binario(df[col]).astype(int)

    df["Cantidad_empresas"] = (
        df["Tipo_cliente_bancoomeva"]
        + df["Tipo_cliente_prepagada"]
        + df["Tipo_cliente_seguros"]
        + df["Tipo_cliente_adicionales"]
    )

    return df[["Id", "Cantidad_empresas"]].drop_duplicates(subset="Id", keep="first")


def calcular_perseverancia_vs_ingresos(v_360: pd.DataFrame, demografica: pd.DataFrame) -> pd.DataFrame:
    """Calcula el ratio valor perseverancia / ingresos (D2).

    Args:
        v_360: DataFrame de ``bases['v_360']`` con columna ``Valorperseverancia``.
        demografica: DataFrame de ``bases['demografica']`` con columna ``Ingresos``.

    Returns:
        DataFrame con columnas ``['Id', 'Valor_perseverancia_vs_ingresos']``.
        Valores negativos quedan en NaN, igual que en el proyecto original.
    """
    perseverancia = (
        v_360
        .assign(Id=lambda d: _id_a_str(d["Identificacion"]))
        [["Id", "Valorperseverancia"]]
        .drop_duplicates(subset="Id", keep="first")
    )
    ingresos = (
        demografica
        .assign(Id=lambda d: _id_a_str(d["ID"]))
        [["Id", "Ingresos"]]
        .drop_duplicates(subset="Id", keep="first")
    )

    df = perseverancia.merge(right=ingresos, how="left", on="Id")

    df["Valor_perseverancia_vs_ingresos"] = (
        df["Valorperseverancia"] / df["Ingresos"].replace(0, np.nan)
    ).astype("Float64").round(3)
    df["Valor_perseverancia_vs_ingresos"] = np.where(
        df["Valor_perseverancia_vs_ingresos"].fillna(0) < 0,
        np.nan,
        df["Valor_perseverancia_vs_ingresos"],
    )

    return df[["Id", "Valor_perseverancia_vs_ingresos"]]


def calcular_cuotas_pagadas_vs_antiguedad(demografica: pd.DataFrame) -> pd.DataFrame:
    """Calcula la proporción de cuotas pagadas vs las esperadas por antigüedad (D4).

    Args:
        demografica: DataFrame de ``bases['demografica']`` con columnas
            ``['Cuotas_canceladas_aportes', 'Antiguedad']``.

    Returns:
        DataFrame con columnas ``['Id', 'Cuotas_pagadas_vs_antiguedad']``,
        clipeada a [0, 1].
    """
    df = (
        demografica
        .assign(Id=lambda d: _id_a_str(d["ID"]))
        [["Id", "Cuotas_canceladas_aportes", "Antiguedad"]]
        .drop_duplicates(subset="Id", keep="first")
    )

    df["Cuotas_pagadas_vs_antiguedad"] = np.where(
        df["Antiguedad"] == 0,
        0,
        (df["Cuotas_canceladas_aportes"] / (df["Antiguedad"] * 12))
        .astype("float32")
        .round(3),
    )
    df["Cuotas_pagadas_vs_antiguedad"] = np.where(
        df["Cuotas_pagadas_vs_antiguedad"] > 1, 1, df["Cuotas_pagadas_vs_antiguedad"]
    )

    return df[["Id", "Cuotas_pagadas_vs_antiguedad"]]


def calcular_distancia_ultima_reactivacion(
    poblacion: pd.DataFrame,
    reactivaciones_historicas: pd.DataFrame,
    demografica: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula la distancia en meses a la última reactivación previa (D3).

    Para cada cédula, busca el periodo más reciente en su historial de
    reactivaciones (siempre anterior al periodo objetivo, ya filtrado en la
    extracción). Si no tiene historial, usa como fallback ``Antiguedad * 12``,
    igual que en el proyecto original.

    Args:
        poblacion: Salida de `_poblacion_normalizada`, columnas ``['Periodo', 'Id']``.
        reactivaciones_historicas: DataFrame de
            ``bases['reactivaciones_historicas']``, columnas ``['Periodo', 'ID']``.
        demografica: DataFrame de ``bases['demografica']`` con columna ``Antiguedad``.

    Returns:
        DataFrame con columnas ``['Id', 'Distancia_ultima_reactivacion']``.
        Cédulas sin historial de reactivación NI dato de `Antiguedad` (no
        cruzaron con `demografica`) quedan en NaN a propósito, no en un valor
        fabricado: el scoring ya sabe ignorar NaN al promediar la dimensión.
    """
    hist = reactivaciones_historicas.rename(columns={"ID": "Id"}).assign(Id=lambda d: _id_a_str(d["Id"]))

    ultima_reactivacion = (
        hist.groupby("Id")["Periodo"].max().reset_index().rename(columns={"Periodo": "Periodo_ultima_reac"})
    )

    df = poblacion.merge(right=ultima_reactivacion, how="left", on="Id")

    df["Distancia_ultima_reactivacion"] = (
        (
            pd.to_datetime(df["Periodo"], format="%Y%m").dt.year
            - pd.to_datetime(df["Periodo_ultima_reac"], format="%Y%m").dt.year
        )
        * 12
        + (
            pd.to_datetime(df["Periodo"], format="%Y%m").dt.month
            - pd.to_datetime(df["Periodo_ultima_reac"], format="%Y%m").dt.month
        )
    )

    antiguedad = (
        demografica
        .assign(Id=lambda d: _id_a_str(d["ID"]))
        [["Id", "Antiguedad"]]
        .drop_duplicates(subset="Id", keep="first")
    )
    df = df.merge(right=antiguedad, how="left", on="Id")

    df["Distancia_ultima_reactivacion"] = df["Distancia_ultima_reactivacion"].fillna(
        df["Antiguedad"] * 12
    )

    return df[["Id", "Distancia_ultima_reactivacion"]]


def calcular_cantidad_reactivaciones_previas(
    poblacion: pd.DataFrame,
    reactivaciones_historicas: pd.DataFrame,
) -> pd.DataFrame:
    """Cuenta las reactivaciones previas al periodo objetivo por cédula (D4).

    Args:
        poblacion: Salida de `_poblacion_normalizada`, columnas ``['Periodo', 'Id']``.
        reactivaciones_historicas: DataFrame de
            ``bases['reactivaciones_historicas']``, columnas ``['Periodo', 'ID']``.

    Returns:
        DataFrame con columnas ``['Id', 'Cantidad_Reactivaciones_Previas']``.
        Cédulas sin historial quedan en 0.
    """
    hist = reactivaciones_historicas.rename(columns={"ID": "Id"}).assign(Id=lambda d: _id_a_str(d["Id"]))

    conteo = hist.groupby("Id").size().reset_index(name="Cantidad_Reactivaciones_Previas")

    df = poblacion[["Id"]].drop_duplicates().merge(right=conteo, how="left", on="Id")
    df["Cantidad_Reactivaciones_Previas"] = df["Cantidad_Reactivaciones_Previas"].fillna(0).astype(int)

    return df


def calcular_recencia_ultimo_producto(
    poblacion: pd.DataFrame,
    tenencia_historica: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula la recencia de la última mejora de tenencia de productos (D3).

    Detecta incrementos consecutivos de ``Numcantidadproductos`` dentro de la
    ventana histórica de 12 meses (ya delimitada en la extracción) y calcula
    la distancia en meses desde el periodo de referencia (mes anterior al
    periodo objetivo, el mismo que usa `extract_tenencia_historica`) hasta la
    última mejora detectada.

    Args:
        poblacion: Salida de `_poblacion_normalizada`, columnas ``['Periodo', 'Id']``.
        tenencia_historica: DataFrame de ``bases['tenencia_historica']``,
            columnas ``['Periodo', 'Id', 'Numcantidadproductos']``.

    Returns:
        DataFrame con columnas ``['Id', 'Recencia_ultimo_producto']``.
        Cédulas sin mejora detectada quedan imputadas con 12.
    """
    hist = tenencia_historica.assign(Id=lambda d: _id_a_str(d["Id"])).sort_values(by=["Id", "Periodo"])

    hist["Producto_rezagado"] = hist.groupby("Id")["Numcantidadproductos"].shift(1)
    hist["Mejora"] = hist["Numcantidadproductos"] > hist["Producto_rezagado"]

    ultima_mejora = (
        hist[hist["Mejora"]]
        .groupby("Id")["Periodo"]
        .max()
        .reset_index()
        .rename(columns={"Periodo": "Periodo_mejora_max"})
    )

    df = poblacion[["Id", "Periodo"]].drop_duplicates(subset="Id").copy()
    df["Periodo_referencia"] = (
        pd.to_datetime(df["Periodo"], format="%Y%m") - pd.DateOffset(months=1)
    ).dt.strftime("%Y%m")
    df = df.merge(right=ultima_mejora, how="left", on="Id")

    df["Recencia_ultimo_producto"] = (
        (
            pd.to_datetime(df["Periodo_referencia"], format="%Y%m").dt.year
            - pd.to_datetime(df["Periodo_mejora_max"], format="%Y%m").dt.year
        )
        * 12
        + (
            pd.to_datetime(df["Periodo_referencia"], format="%Y%m").dt.month
            - pd.to_datetime(df["Periodo_mejora_max"], format="%Y%m").dt.month
        )
    )
    df["Recencia_ultimo_producto"] = df["Recencia_ultimo_producto"].fillna(12).astype(int)

    return df[["Id", "Recencia_ultimo_producto"]]


def calcular_clv(clv: pd.DataFrame) -> pd.DataFrame:
    """Extrae el rango de CLV potencial (D4).

    Args:
        clv: DataFrame de ``bases['clv']`` con columnas
            ``['Identificacion', 'Rango_CLV_Potencial']``. ``Rango_CLV_Potencial``
            tiene formato ``N. Etiqueta``; se extrae solo la etiqueta después
            del primer punto.

    Returns:
        DataFrame con columnas ``['Id', 'Clv']``.
    """
    df = clv[["Identificacion", "Rango_CLV_Potencial"]].copy()
    df["Rango_CLV_Potencial"] = df["Rango_CLV_Potencial"].str.split(".", n=-1).str[1].str.strip()
    df["Id"] = _id_a_str(df["Identificacion"])

    return (
        df.rename(columns={"Rango_CLV_Potencial": "Clv"})[["Id", "Clv"]]
        .drop_duplicates(subset="Id", keep="first")
    )


def calcular_saldoaportes(demografica: pd.DataFrame) -> pd.DataFrame:
    """Extrae el saldo de aportes del asociado (D4). Literal, sin transformación.

    Args:
        demografica: DataFrame de ``bases['demografica']`` con columna ``Saldoaportes``.

    Returns:
        DataFrame con columnas ``['Id', 'Saldoaportes']``.
    """
    return (
        demografica
        .assign(Id=lambda d: _id_a_str(d["ID"]))
        [["Id", "Saldoaportes"]]
        .drop_duplicates(subset="Id", keep="first")
    )
    
def cambios_sipas(sipas:pd.DataFrame) -> pd.DataFrame:
    
    # --- Se calcula :Perseverancia_Cerca ---#
    
    df = sipas.copy()
    df = (
        df
        .assign(
            Perseverancia_Cerca = lambda df:
                np.where(
                    df['Meses_Hasta_Perseverancia'].clip(0) <= 60, 
                    1, 
                    0
                )
        )
        .drop(
            columns={
                'Meses_Hasta_Perseverancia'
            }
        )
    )
    
    return df 
    
def calcular_promedio_fac_rec(
    df: pd.DataFrame,
    fac_rec: pd.DataFrame,
    meses: int = 6,
    nombre: str = "GECC",
) -> pd.DataFrame:
    """Calcula recaudo y factura promedio de los últimos `meses` meses (D4).

    Ventana retrospectiva única de `meses` meses sobre el histórico de
    facturación/recaudo (``bases['fac_rec']``). Versión simplificada: a
    diferencia de una ventana móvil genérica multi-métrica, solo conserva
    recaudo y factura promedio para una única fuente/ventana.

    Args:
        df: DataFrame base con columnas ``['Periodo', 'Id']`` (normalmente la
            salida de `_poblacion_normalizada`). ``Periodo`` en formato
            ``YYYYMM`` (str o int).
        fac_rec: Histórico de facturación/recaudo (``bases['fac_rec']``) con
            columnas ``['Periodo', 'Id', 'Valor_Recaudado_Total', 'Cuota_Mes',
            'Vencido_Mes']``. No se muta el DataFrame original.
        meses: Tamaño de la ventana retrospectiva en meses. Default 6.
        nombre: Prefijo que identifica la fuente (e.g. ``'GECC'``).

    Returns:
        DataFrame con columnas ``['Periodo', 'Id',
        f'Factura_Total_Promedio_{nombre}_{meses}M',
        f'Recaudo_Total_Promedio_{nombre}_{meses}M',
        f'Vencido_Promedio_{nombre}_{meses}M',
        f'Pct_Mora_{nombre}_{meses}M']``. Cédulas sin transacciones en la
        ventana quedan en 0.

        ``Vencido_Mes`` (saldo vencido, un *stock* acumulado a la fecha del
        estado de cuenta) y ``Cuota_Mes`` (lo facturado ese mes puntual, un
        *flujo*) no se pueden sumar sin más a lo largo de varios meses: el
        vencido de hoy ya incluye las cuotas de meses anteriores que quedaron
        sin pagar, así que sumar `meses` cuotas y además sumarle el vencido
        duplicaría esa plata. Por eso:

        - ``Factura_Total_Promedio`` es puramente el promedio de ``Cuota_Mes``
          en la ventana (flujo, sin vencido mezclado — cero riesgo de doble
          conteo).
        - ``Vencido_Promedio`` es el promedio simple de los `meses` valores
          mensuales reales de ``Vencido_Mes`` (stock, columna independiente).
        - ``Pct_Mora`` = ``Vencido_Promedio / (Vencido_Promedio +
          Factura_Total_Promedio)``: el vencido entra una sola vez (no se
          re-suma mes a mes), como fracción de "lo que debe vencido + lo que
          factura normalmente". Es la señal de severidad recomendada, no
          ``Vencido_Promedio`` ni ``Factura_Total`` en pesos absolutos — un
          socio grande factura más y por lo tanto también puede acumular un
          vencido más grande en pesos aunque esté proporcionalmente más al
          día que uno pequeño. Cédulas sin factura ni vencido en la ventana
          quedan en Pct_Mora = 0 (no hay nada que medir).
    """
    df = df[["Periodo", "Id"]].assign(Id=lambda d: _id_a_str(d["Id"])).copy()
    fac_rec = fac_rec.rename(columns={"Periodo": "Periodo_Insumo"}).assign(Id=lambda d: _id_a_str(d["Id"]))

    df_merge = df.merge(right=fac_rec, how="left", on="Id")
    df_merge["Periodo_dt"] = pd.to_datetime(df_merge["Periodo"].astype(str), format="%Y%m")
    df_merge["Periodo_Insumo_dt"] = pd.to_datetime(df_merge["Periodo_Insumo"].astype(str), format="%Y%m")

    ventana = df_merge[
        df_merge["Periodo_Insumo_dt"].between(
            df_merge["Periodo_dt"] - pd.DateOffset(months=meses),  # type: ignore[operator]
            df_merge["Periodo_dt"],
            inclusive="left",
        )
    ].copy()

    col_rec = f"Recaudo_Total_Promedio_{nombre}_{meses}M"
    col_fac = f"Factura_Total_Promedio_{nombre}_{meses}M"
    col_venc = f"Vencido_Promedio_{nombre}_{meses}M"
    col_pct_mora = f"Pct_Mora_{nombre}_{meses}M"

    ventana[col_rec] = ventana.groupby(["Id", "Periodo"])["Valor_Recaudado_Total"].transform("sum") / meses
    ventana[col_fac] = ventana.groupby(["Id", "Periodo"])["Cuota_Mes"].transform("sum") / meses
    ventana[col_venc] = ventana.groupby(["Id", "Periodo"])["Vencido_Mes"].transform("mean")

    condiciones = [ventana[col_rec] < 0, ventana[col_rec] > ventana[col_fac]]
    valores = [0, ventana[col_fac]]
    ventana[col_rec] = np.select(condlist=condiciones, choicelist=valores, default=ventana[col_rec])

    denominador_mora = ventana[col_venc] + ventana[col_fac]
    ventana[col_pct_mora] = np.where(
        denominador_mora > 0, ventana[col_venc] / denominador_mora, 0
    )

    resultado = (
        ventana[["Periodo", "Id", col_fac, col_rec, col_venc, col_pct_mora]]
        .fillna(0)
        .drop_duplicates(subset=["Id", "Periodo"])
    )

    return df.merge(right=resultado, how="left", on=["Id", "Periodo"]).fillna(
        {col_fac: 0, col_rec: 0, col_venc: 0, col_pct_mora: 0}
    )


def calcular_intenciones_retiro(
    poblacion: pd.DataFrame,
    consolidado_reactivaciones: pd.DataFrame,
    meses: int = 12,
) -> pd.DataFrame:
    """Cuenta las intenciones de retiro registradas en el último año por cédula.

    Filtra ``bases['consolidado_reactivaciones']`` (informe de retención/
    reconexión) a los registros de cada cédula con ``anomes`` dentro de los
    `meses` meses anteriores al periodo objetivo (ventana retrospectiva, sin
    incluir el propio periodo objetivo — ej. Periodo 202606 -> ventana
    [202506, 202606)), y cuenta cuántas veces aparece cada cédula en esa
    ventana (cada aparición es un intento de retiro/formulario de retención
    gestionado).

    Args:
        poblacion: Salida de `_poblacion_normalizada`, columnas ``['Periodo', 'Id']``.
        consolidado_reactivaciones: DataFrame de
            ``bases['consolidado_reactivaciones']`` con columnas
            ``['Número de Identificación', 'anomes']`` (entre otras, no usadas
            aquí). Puede tener varias filas por cédula y mes.
        meses: Tamaño de la ventana retrospectiva en meses. Default 12.

    Returns:
        DataFrame con columnas ``['Id', 'Periodo', 'intenciones_retiro_1y']``.
        Cédulas sin registros en la ventana quedan en 0.
    """
    df = poblacion[["Periodo", "Id"]].assign(Id=lambda d: _id_a_str(d["Id"])).copy()

    reg = (
        consolidado_reactivaciones
        .rename(columns={"Número de Identificación": "Id"})
        .dropna(subset=["Id"])
        .assign(Id=lambda d: _id_a_str(d["Id"]))
        [["Id", "anomes"]]
    )

    df_merge = df.merge(right=reg, how="left", on="Id")
    df_merge["Periodo_dt"] = pd.to_datetime(df_merge["Periodo"].astype(str), format="%Y%m")
    df_merge["anomes_dt"] = pd.to_datetime(df_merge["anomes"].astype("Int64").astype(str), format="%Y%m")

    ventana = df_merge[
        df_merge["anomes_dt"].between(
            df_merge["Periodo_dt"] - pd.DateOffset(months=meses),  # type: ignore[operator]
            df_merge["Periodo_dt"],
            inclusive="left",
        )
    ]

    conteo = (
        ventana
        .groupby(["Id", "Periodo"])
        .size()
        .reset_index(name="intenciones_retiro_1y")
    )

    return (
        df.merge(right=conteo, how="left", on=["Id", "Periodo"])
        .fillna({"intenciones_retiro_1y": 0})
        .astype({"intenciones_retiro_1y": int})
    )


def calcular_tiempo_inactividad(features_inactivos: pd.DataFrame) -> pd.DataFrame:
    """Extrae el tiempo de inactividad en meses (D1). Literal, sin transformación.

    Args:
        features_inactivos: DataFrame de ``bases['features_inactivos']`` con
            columnas ``['CEDULA', 'TIEMPO INACTIVO EN MESES']``.

    Returns:
        DataFrame con columnas ``['Id', 'Tiempo_Inactividad_Meses']``.
    """
    return (
        features_inactivos
        .assign(Id=lambda d: _id_a_str(d["CEDULA"]))
        .rename(columns={"TIEMPO INACTIVO EN MESES": "Tiempo_Inactividad_Meses"})
        [["Id", "Tiempo_Inactividad_Meses"]]
        .drop_duplicates(subset="Id", keep="first")
    )


def calcular_oferta_reactivacion(features_inactivos: pd.DataFrame) -> pd.DataFrame:
    """Extrae la cantidad de ofertas de reactivación disponibles (D1), rango [0, 4].

    Args:
        features_inactivos: DataFrame de ``bases['features_inactivos']`` con
            columnas ``['CEDULA', 'OFERTA_DISPONIBLE']``.

    Returns:
        DataFrame con columnas ``['Id', 'Oferta_Reactivacion_Disponible']``.
    """
    return (
        features_inactivos
        .assign(Id=lambda d: _id_a_str(d["CEDULA"]))
        .rename(columns={"OFERTA_DISPONIBLE": "Oferta_Reactivacion_Disponible"})
        [["Id", "Oferta_Reactivacion_Disponible"]]
        .drop_duplicates(subset="Id", keep="first")
    )


def calcular_valor_capitalizado(sipas: pd.DataFrame) -> pd.DataFrame:
    """Extrae el valor capitalizado en plan básico (D2). Literal, sin transformación.

    Los nulos representan cédulas sin plan básico (no aplica), no dato
    faltante: se preservan como NaN a propósito, no se imputan.

    Args:
        features_inactivos: DataFrame de ``bases['features_inactivos']`` con
            columnas ``['CEDULA', 'Valor Capitalizado']``.

    Returns:
        DataFrame con columnas ``['Id', 'Valor_Capitalizado']``.
    """
    return (
        sipas
        .assign(Id=lambda d: _id_a_str(d[""]))
        [["Id", "Valor_Capitalizado"]]
        .drop_duplicates(subset="Id", keep="first")
    )


def calcular_tiempo_perseverancia(sipas: pd.DataFrame) -> pd.DataFrame:
    """Extrae el indicador booleano de cercanía a perseverar (D4).

    Los nulos representan cédulas sin plan básico (no aplica, misma población
    que en `calcular_valor_capitalizado`), no dato faltante: se preservan como
    NaN a propósito, no se imputan. Ya vive en {0, 1}: no requiere escalarse.

    Args:
        features_inactivos: DataFrame de ``bases['features_inactivos']`` con
            columnas ``['CEDULA', 'Perseverancia Cerca']``.

    Returns:
        DataFrame con columnas ``['Id', 'Perseverancia_Cerca']``.
    """
    return (
        sipas
        .assign(Id=lambda d: _id_a_str(d["ID"]))
        [["Id", "Perseverancia_Cerca"]]
        .drop_duplicates(subset="Id", keep="first")
    )


def calcular_n_usos(enriquecimiento_360: pd.DataFrame) -> pd.DataFrame:
    """Extrae la cantidad total de eventos/usos del asociado (D2). Literal.

    Args:
        enriquecimiento_360: DataFrame de ``bases['enriquecimiento_360']`` con
            columnas ``['Identificacion', 'Total_Eventos']``.

    Returns:
        DataFrame con columnas ``['Id', 'Total_Eventos']``.
    """
    return (
        enriquecimiento_360
        .assign(Id=lambda d: _id_a_str(d["Identificacion"]))
        [["Id", "Total_Eventos"]]
        .drop_duplicates(subset="Id", keep="first")
    )


def calcular_pqr(pqrs: pd.DataFrame) -> pd.DataFrame:
    """Extrae la cantidad de PQR del último año (D3). Literal, sin transformación.

    Args:
        enriquecimiento_360: DataFrame de ``bases['enriquecimiento_360']`` con
            columnas ``['Identificacion', 'Cantidad_pqr_ultimo_anno']``.

    Returns:
        DataFrame con columnas ``['Id', 'Cantidad_pqr_ultimo_anno']``.
    """
    return (
        pqrs
        .assign(Id=lambda d: _id_a_str(d["Identificacion"]))
        [["Id", "Cantidad_pqr_ultimo_anno"]]
        .drop_duplicates(subset="Id", keep="first")
    )


def calcular_turnos_oficina(v_360: pd.DataFrame) -> pd.DataFrame:
    """Extrae los turnos en oficina en los últimos 12 meses (D3).

    Args:
        v_360: DataFrame de ``bases['v_360']`` con columna
            ``Turnos_En_Oficinas_Total_Ult12Meses``.

    Returns:
        DataFrame con columnas ``['Id', 'Turnos_En_Oficinas_Total_Ult12Meses']``.
        Nulos imputados a 0 (cédula sin turnos registrados).
    """
    return (
        v_360
        .assign(Id=lambda d: _id_a_str(d["Identificacion"]))
        [["Id", "Turnos_En_Oficinas_Total_Ult12Meses"]]
        .drop_duplicates(subset="Id", keep="first")
        .fillna({"Turnos_En_Oficinas_Total_Ult12Meses": 0})
    )


def calcular_alertas_externas(enriquecimiento_360: pd.DataFrame) -> pd.DataFrame:
    """Extrae las 3 alertas binarias del sector externo (D5). Literal, sin transformación.

    Args:
        enriquecimiento_360: DataFrame de ``bases['enriquecimiento_360']`` con
            columnas ``['Identificacion', 'Alerta_Habito_Pago_Externo',
            'Alerta_Estado_Creditos_Externos', 'Alerta_Capacidad_Pago_Externo']``
            (0/1, sin nulos).

    Returns:
        DataFrame con columnas ``['Id', 'Alerta_Habito_Pago_Externo',
        'Alerta_Estado_Creditos_Externos', 'Alerta_Capacidad_Pago_Externo']``.
    """
    return (
        enriquecimiento_360
        .assign(Id=lambda d: _id_a_str(d["Identificacion"]))
        [["Id", "Alerta_Habito_Pago_Externo", "Alerta_Estado_Creditos_Externos",
          "Alerta_Capacidad_Pago_Externo"]]
        .drop_duplicates(subset="Id", keep="first")
    )

# ─── Orquestadora ─────────────────────────────────────────────────────────────

def build_analytic_score(bases: dict[str, pd.DataFrame], analytic_path: str | None = None) -> pd.DataFrame:
    """Arma la base analítica de scoring D1-D5 a partir de las bases crudas.

    Función de entrada principal del módulo. Toma el diccionario que entrega
    `extract_raw` y calcula las variables de scoring D1-D5 más Periodo/Id,
    exportando el resultado a Parquet.

    Args:
        bases: Diccionario retornado por `extract_raw`. Debe contener las claves
            ``poblacion_inactivos``, ``cantidad_productos``, ``tenencia_historica``,
            ``v_360``, ``demografica``, ``reactivaciones_historicas``, ``clv``,
            ``fac_rec``, ``consolidado_reactivaciones``, ``features_inactivos``,
            ``enriquecimiento_360``.
        analytic_path: Ruta de salida personalizada. Si es None, exporta a
            ``data/analytic/analytic_score_base.parquet``.

    Returns:
        DataFrame con columnas ``['Periodo', 'Id',
        'Tiempo_Inactividad_Meses', 'Factura_Total_Promedio_GECC_6M',
        'Recaudo_Total_Promedio_GECC_6M', 'Oferta_Reactivacion_Disponible',
        'intenciones_retiro_1y', 'Numcantidadproductos', 'Cantidad_empresas',
        'Valor_perseverancia_vs_ingresos', 'Total_Eventos', 'Valor_Capitalizado',
        'Distancia_ultima_reactivacion', 'Recencia_ultimo_producto',
        'Turnos_En_Oficinas_Total_Ult12Meses', 'Cantidad_pqr_ultimo_anno',
        'Cuotas_pagadas_vs_antiguedad', 'Cantidad_Reactivaciones_Previas',
        'Clv', 'Saldoaportes', 'Perseverancia_Cerca',
        'Alerta_Habito_Pago_Externo', 'Alerta_Estado_Creditos_Externos',
        'Alerta_Capacidad_Pago_Externo']``.
    """
    print("Iniciando transformación (base analítica D1-D5)...")

    poblacion = _poblacion_normalizada(bases["poblacion_inactivos"])

    df = poblacion.copy()

    # D1 · SEVERIDAD INACTIVIDAD
    df = df.merge(right=calcular_tiempo_inactividad(bases["features_inactivos"]), how="left", on="Id")
    df = df.merge(
        right=calcular_promedio_fac_rec(poblacion, bases["fac_rec"]),
        how="left", on=["Id", "Periodo"],
    )
    df = df.merge(right=calcular_oferta_reactivacion(bases["features_inactivos"]), how="left", on="Id")
    df = df.merge(
        right=calcular_intenciones_retiro(poblacion, bases["consolidado_reactivaciones"]),
        how="left", on=["Id", "Periodo"],
    )

    # D2 · ENGANCHE
    df = df.merge(right=calcular_numcantidadproductos(bases["cantidad_productos"]), how="left", on="Id")
    df = df.merge(right=calcular_cantidad_empresas(bases["v_360"]), how="left", on="Id")
    df = df.merge(
        right=calcular_perseverancia_vs_ingresos(bases["v_360"], bases["demografica"]),
        how="left", on="Id",
    )
    df = df.merge(right=calcular_n_usos(bases["enriquecimiento_360"]), how="left", on="Id")
    df = df.merge(right=calcular_valor_capitalizado(bases["sipas"]), how="left", on="Id")

    # D3 · RECENCIA
    df = df.merge(
        right=calcular_distancia_ultima_reactivacion(
            poblacion, bases["reactivaciones_historicas"], bases["demografica"]
        ),
        how="left", on="Id",
    )
    df = df.merge(
        right=calcular_recencia_ultimo_producto(poblacion, bases["tenencia_historica"]),
        how="left", on="Id",
    )
    df = df.merge(right=calcular_turnos_oficina(bases["v_360"]), how="left", on="Id")
    df = df.merge(right=calcular_pqr(bases["pqrs"]), how="left", on="Id")

    # D4 · VÍNCULO HISTÓRICO
    df = df.merge(right=calcular_cuotas_pagadas_vs_antiguedad(bases["demografica"]), how="left", on="Id")
    df = df.merge(
        right=calcular_cantidad_reactivaciones_previas(poblacion, bases["reactivaciones_historicas"]),
        how="left", on="Id",
    )
    df = df.merge(right=calcular_clv(bases["clv"]), how="left", on="Id")
    df = df.merge(right=calcular_saldoaportes(bases["demografica"]), how="left", on="Id")
    df = df.merge(right=calcular_tiempo_perseverancia(bases["sipas"]), how="left", on="Id")

    # D5 · EXTERNO
    df = df.merge(right=calcular_alertas_externas(bases["enriquecimiento_360"]), how="left", on="Id")

    df["Numcantidadproductos"] = df["Numcantidadproductos"].fillna(0)
    df["Cantidad_empresas"] = df["Cantidad_empresas"].fillna(0)

    path_salida = Path(analytic_path) if analytic_path else Path(__file__).parents[2] / "data" / "analytic"
    path_salida.mkdir(parents=True, exist_ok=True)

    archivo_salida = path_salida / "analytic_score_base.parquet"
    df.to_parquet(archivo_salida, engine="pyarrow", index=False)
    print(f"Base analítica de scoring -- Exportada en {archivo_salida}")

    return df
