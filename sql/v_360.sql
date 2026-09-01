SELECT
	Periodo as [Periodo],
	Identificacion as [Identificacion],
	Tipo_Cliente_Bancoomeva as [Tipo_cliente_bancoomeva],
	Tipo_Cliente_Prepagada as [Tipo_cliente_prepagada],
	Tipo_Cliente_Seguros as [Tipo_cliente_seguros],
	Tipo_Cliente_Adicionales as [Tipo_cliente_adicionales],
	ValorPerseverancia as [Valorperseverancia],
	Turnos_En_Oficinas_Total_Ult12Meses as [Turnos_En_Oficinas_Total_Ult12Meses],
	(CantidadEventosEducacion_Histo + 
	CantidadEventosRecreacion_Histo + 
	CantidadEventosFundacion_Histo) AS [Total_Eventos],
	Alerta_Habito_Pago_Externo AS [Alerta_Habito_Pago_Externo],
	Alerta_Estado_Creditos_Externos AS [Alerta_Estado_Creditos_Externos],
	Alerta_Capacidad_Pago_Externo AS [Alerta_Capacidad_Pago_Externo]
FROM Operaciones.dbo.ConsultaIntegral360_Diaria
WHERE
	Operaciones.$partition.pf_mes(dtmFechaInsercion) = Operaciones.$partition.pf_mes('?')
	AND Identificacion IN ({ids})