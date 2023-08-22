if __name__ == '__main__':
    from evaluate_DT import evaluate

    evaluate(DT_model="TobiTob/decision_transformer_merged3", buildings_to_use="validation",
             TR=-0, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_resultsR_0000.txt')

    evaluate(DT_model="TobiTob/decision_transformer_merged3", buildings_to_use="validation",
             TR=-3000, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_resultsR_3000.txt')

    evaluate(DT_model="TobiTob/decision_transformer_merged3", buildings_to_use="validation",
             TR=-6000, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_resultsR_6000.txt')

    evaluate(DT_model="TobiTob/decision_transformer_merged3", buildings_to_use="validation",
             TR=-9000, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_resultsR_9000.txt')

    evaluate(DT_model="TobiTob/decision_transformer_merged3", buildings_to_use="validation",
             TR=-12000, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_resultsR_12000.txt')

    evaluate(DT_model="TobiTob/decision_transformer_merged3", buildings_to_use="validation",
             TR=-15000, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_resultsR_15000.txt')

    evaluate(DT_model="TobiTob/decision_transformer_merged3", buildings_to_use="validation",
             TR=-30000, evaluation_interval=168, simulation_start_end=[0, 8759], file_to_save='_resultsR_30000.txt')

