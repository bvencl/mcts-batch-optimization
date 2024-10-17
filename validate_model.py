from factory import *
from utils.utils import *
from utils.validate import validate_model

def main():
    args = get_args()
    config = read(args.config)
    config.data_loader.custom_sampler = 0
    
    file = config.paths.model_path + config.paths.model_name + ".pth"
    
    _, val_dataset, test_dataset = DatasetFactory.create(config=config)
    _, val_loader, _ = DataLoaderFactory.create(config=config, train_dataset=_, val_dataset=val_dataset, test_dataset=test_dataset)
    
    my_model = ModelFactory.create(config=config)
    my_criterion, _, _ = AgentFactory.create(config=config, model=my_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_model.to(device)
    val_loss , val_acc = validate_model(my_model, data_loader=val_loader, criterion=my_criterion)
    
    print(f'Validation accuracy: {val_acc}')
    print(f'Validation loss: {val_loss}')
    
if __name__ == "__main__":
    main()