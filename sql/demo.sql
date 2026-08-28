WITH FacSegmentos AS (
                         SELECT
                            strPeriodo AS Periodo,
                            strIdentificacion AS ID,
                            numCantidadAniosAntiguedad as Antiguedad
                         FROM BodegaCorporativa.bodega.factSegmentos
                         WHERE BodegaCorporativa.$partition.pf_mes(dtmFechaInsercion) = BodegaCorporativa.$partition.pf_mes('?')
                    ),
    Intenciones_retiro AS(
                        SELECT
                            strIdentificacion, 
                            COUNT(*) AS [intenciones_retiro_1y]
                        FROM [Operaciones].[dbo].[intencionesDeRetiro]
                        WHERE 
                            Periodo BETWEEN ? AND ?
                        GROUP BY 
                            strIdentificacion
        )
                    SELECT
                        dem.strPeriodo as Periodo,
                        dem.Documento as ID,
                        dem.Ingresos,
                        dem.Cuotas_canceladas_aportes,
                        dem.Saldo_aportes as Saldoaportes,
                        seg.Antiguedad, 
                        ir.intenciones_retiro_1y
                    FROM BodegaCorporativa.Conocimiento.v_Demografica Dem
                    INNER JOIN FacSegmentos seg
                        ON dem.strPeriodo = seg.Periodo
                        AND dem.Documento = seg.ID
					LEFT JOIN Intenciones_retiro ir
					ON
						Dem.Documento = ir.strIdentificacion
                    WHERE BodegaCorporativa.$partition.pf_mes(dem.dtmFechaInsercion) = BodegaCorporativa.$partition.pf_mes('?')
                        AND dem.Documento IN ({ids})
