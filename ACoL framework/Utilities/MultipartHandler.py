import copy
from spade.message import Message

class MultipartHandler:

    def __init__(self) -> None:
        self.multipart_message_storage: dict[str,str] = {}

    def is_multipart(self, message: Message) -> bool:
        return message.body.startswith('multipart#')
    
    def any_multipart_waiting(self) -> bool:
        return len(self.multipart_message_storage.keys()) > 0

    def is_multipart_complete(self, message: Message) -> bool:
        sender = str(message.sender).split("@")[0]
        if not sender in self.multipart_message_storage.keys():
            return None
        for part in self.multipart_message_storage[sender]:
            if part is None:
                return False
        return True
    
    def rebuild_multipart_content(self, parts: list[str]) -> str:
        content = ""
        for part in parts:
            content += part
        return content

    def rebuild_multipart(self, message: Message) -> Message:
        # multipart_meta = message.get_metadata("multipart")
        if self.is_multipart(message):
            sender = str(message.sender).split("@")[0]
            multipart_meta = message.body.split('|')[0]
            multipart_meta_parts = multipart_meta.split('#')[1]
            part_number = int(multipart_meta_parts.split("/")[0])
            total_parts = int(multipart_meta_parts.split("/")[1])
            if not sender in self.multipart_message_storage.keys():
                self.multipart_message_storage[sender] = [None] * total_parts
            self.multipart_message_storage[sender][part_number - 1] = message.body[len(multipart_meta + '|'):]
            if self.is_multipart_complete(message):
                message.body = self.rebuild_multipart_content(self.multipart_message_storage[sender])
                del self.multipart_message_storage[sender]
                return message
        return None
    
    def divide_content(self, content: str, size: int) -> list[str]:
        return [content[i:i+size] for i in range(0, len(content), size)]  
    
    def multipart_content(self, content: str, max_size: int) -> list[str]:
        if len(content) > max_size:
            multiparts = self.divide_content(content, max_size - len("multipart#9999/9999|"))
            return [f"multipart#{i + 1}/{len(multiparts)}|" + part.strip() for i, part in enumerate(multiparts)]
        return None
    
    def generate_multipart_messages(self, content: str, max_size: int, message_base: Message) -> list[Message]:
        multiparts = self.multipart_content(content, max_size)
        if multiparts is not None:
            multiparts_messages = []
            for multipart in multiparts:
                message = copy.deepcopy(message_base)
                message.body = multipart
                multiparts_messages.append(message)
            return multiparts_messages
        return None