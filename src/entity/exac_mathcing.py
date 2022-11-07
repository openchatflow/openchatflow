class ExactMatchingParams:
    def __init__(self, param_list:list, param_names:str):
        self.parameters = param_list
        self.label = param_names
    

    def replace_entities(self, text):
        entity_replaced_text = text
        entity = []

        for p in self.parameters:
            if p in text:
                entity_replaced_text = entity_replaced_text.replace(f"<{self.labels}>", p)
                entity.append(p)
        
        return entity_replaced_text