from utils.utils import read_json

class AnnotationReader:
    def __init__(self, dir_path: str):
        self.tracks = read_json(f'{dir_path}/annotations.json')[0]['tracks']
        self.frames_with_labels = {}

        for track in self.tracks:
            for shape in track['shapes']:
                if not shape['outside']:
                    self.frames_with_labels[shape['frame']] = self._get_class_by_label(track['label'])

    def get_frame_label(self, frame: int):
        return 'no_action' if frame not in self.frames_with_labels else self.frames_with_labels[frame]

    def _get_class_by_label(self, label: str) -> str:
        match label:
            case 'Głowa lewą ręką' | 'Głowa prawą ręką':
                return 'head'
            case 'Korpus lewą ręką' | 'Korpus prawą ręką':
                return 'corpus'
            case 'Blok lewą ręką' | 'Blok prawą ręką':
                return 'block'
            case 'Chybienie lewą ręką' | 'Chybienie prawą ręką':
                return 'no_action'
            case _:
                raise Exception(f'Unexpected label {label}')