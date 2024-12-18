import json
import re
from collections import Counter


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        elif self.dataset_name == 'wsi_report':
            self.clean_report = self.clean_report_pathology
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report
    
    def clean_report_pathology(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
    
class ModernTokenizer(object):
    """A wrapper class that can use different modern tokenizers while maintaining compatibility
    with your existing codebase."""
    
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        
        # Option 1: Medical-specific tokenizer (BioClinicalBERT)
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        # Option 2: General purpose tokenizer (RoBERTa)
        # self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        # Option 3: Domain-adapted tokenizer (PubMedBERT)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        
        # Add special tokens specific to medical reports if needed
        special_tokens = {
            "additional_special_tokens": [
                "<impression>", "</impression>",
                "<findings>", "</findings>",
                "<comparison>", "</comparison>",
                "<indication>", "</indication>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load your existing vocabulary for compatibility
        self.ann = json.loads(open(args.ann_path, 'r').read())
        
        # Map special tokens
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        
        # Create reverse mapping
        self.token2idx = self.tokenizer.get_vocab()
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        
    def __call__(self, report: str) -> List[int]:
        """Convert report text to token ids."""
        # Use the tokenizer's built-in encoding
        encoding = self.tokenizer.encode(
            report,
            add_special_tokens=True,
            max_length=self.args.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding[0].tolist()  # Convert tensor to list
    
    def decode(self, ids: List[int]) -> str:
        """Convert token ids back to text."""
        # Use the tokenizer's built-in decoding
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.post_process_medical_text(text)

    def post_process_medical_text(self, text: str) -> str:
        """Apply domain-specific post-processing to medical reports."""
        # Capitalize anatomical terms
        anatomical_terms = {'chest', 'heart', 'lungs', 'pleural', 'mediastinum'}
        for term in anatomical_terms:
            text = text.replace(f' {term} ', f' {term.capitalize()} ')
        
        # Fix common medical abbreviations
        abbreviations = {
            'pa': 'PA',
            'ap': 'AP',
            'lat': 'LAT',
            'iv': 'IV',
            'ct': 'CT'
        }
        for abbr, replacement in abbreviations.items():
            text = text.replace(f' {abbr} ', f' {replacement} ')
        
        # Fix measurements spacing - using a safer regex pattern
        text = re.sub(r'(\d+)(mm|cm|inches)', r'\1 \2', text)
        
        # Proper sentence capitalization
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences if s]
        text = '. '.join(sentences)
        
        return text.strip()
    
    def decode_batch(self, ids_batch: List[List[int]]) -> List[str]:
        """Decode a batch of token ids to texts."""
        return [self.decode(ids) for ids in ids_batch]
    
    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.token2idx)
    
    def get_token_by_id(self, id: int) -> str:
        """Get token string from token id."""
        return self.idx2token.get(id, self.unk_token)
    
    def get_id_by_token(self, token: str) -> int:
        """Get token id from token string."""
        return self.token2idx.get(token, self.token2idx[self.unk_token])
        
    def batch_encode(self, reports: List[str], 
                    max_length: Optional[int] = None) -> torch.Tensor:
        """Encode a batch of reports to token ids."""
        if max_length is None:
            max_length = self.args.max_seq_length
            
        encodings = self.tokenizer.batch_encode_plus(
            reports,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encodings['input_ids']