import { GPT2Tokenizer } from "../../../src/tokenizers.js";
import { BASE_TEST_STRINGS, SENTENCEPIECE_TEST_STRINGS } from "../test_strings.js";

export const TOKENIZER_CLASS = GPT2Tokenizer;
export const TEST_CONFIG = {
  // - clean_up_tokenization_spaces=true
  // - default pretokenization regex
  "Xenova/gpt2": {
    SIMPLE: {
      text: BASE_TEST_STRINGS.SIMPLE,
      tokens: ["How", "\u0120are", "\u0120you", "\u0120doing", "?"],
      ids: [2437, 389, 345, 1804, 30],
      decoded: "How are you doing?",
    },
    SIMPLE_WITH_PUNCTUATION: {
      text: BASE_TEST_STRINGS.SIMPLE_WITH_PUNCTUATION,
      tokens: ["You", "\u0120should", "'ve", "\u0120done", "\u0120this"],
      ids: [1639, 815, 1053, 1760, 428],
      decoded: "You should've done this",
    },
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["01", "23", "45", "67", "89", "Ġ0", "Ġ1", "Ġ2", "Ġ3", "Ġ4", "Ġ5", "Ġ6", "Ġ7", "Ġ8", "Ġ9", "Ġ10", "Ġ100", "Ġ1000"],
      ids: [486, 1954, 2231, 3134, 4531, 657, 352, 362, 513, 604, 642, 718, 767, 807, 860, 838, 1802, 8576],
      decoded: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "\u0120company", "\u0120was", "\u0120founded", "\u0120in", "\u01202016", "."],
      ids: [464, 1664, 373, 9393, 287, 1584, 13],
      decoded: "The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "\u010a", "'ll", "\u0120!!", "to", "?'", "d", "''", "d", "\u0120of", ",", "\u0120can", "'t", "."],
      ids: [32, 198, 1183, 37867, 1462, 8348, 67, 7061, 67, 286, 11, 460, 470, 13],
      decoded: "A\n'll!!to?'d''d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def", "\u0120main", "():", "\u010a", "\u0109", "pass"],
      ids: [4299, 1388, 33529, 198, 197, 6603],
      decoded: "def main():\n\tpass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".", "to", "String", "();", "\u010a", "to", "String", "();"],
      ids: [1616, 257, 796, 26181, 13, 1462, 10100, 9783, 198, 1462, 10100, 9783],
      decoded: "let a = obj.toString();\ntoString();",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["This", "\u010a", "\u010a", "is", "\u010a", "a", "\u010a", "test", "."],
      ids: [1212, 198, 198, 271, 198, 64, 198, 9288, 13],
      decoded: "This\n\nis\na\ntest.",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "want", "\u00c3\u00a9", "d", ",", "running"],
      ids: [4944, 42949, 2634, 67, 11, 20270],
      decoded: "UNwant\u00e9d,running",
    },
    CONTROL_TOKENS: {
      text: BASE_TEST_STRINGS.CONTROL_TOKENS,
      tokens: ["1", "\u0100", "2", "\u00ef\u00bf\u00bd", "3"],
      ids: [16, 188, 17, 4210, 18],
      decoded: "1\u00002\ufffd3",
    },
    HELLO_WORLD_TITLECASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_TITLECASE,
      tokens: ["Hello", "\u0120World"],
      ids: [15496, 2159],
      decoded: "Hello World",
    },
    HELLO_WORLD_LOWERCASE: {
      text: BASE_TEST_STRINGS.HELLO_WORLD_LOWERCASE,
      tokens: ["hello", "\u0120world"],
      ids: [31373, 995],
      decoded: "hello world",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141", "\u00e6", "\u00b4", "\u00bb", "\u00e7\u013c\u0126", "\u00e7\u013e", "\u0141", "\u00e8", "\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [37955, 162, 112, 119, 21410, 40367, 253, 164, 108, 249, 42468],
      decoded: "\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u0120", "\u0120", "\u0120leading", "\u0120space"],
      ids: [220, 220, 3756, 2272],
      decoded: "   leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["tra", "iling", "\u0120space", "\u0120", "\u0120", "\u0120"],
      ids: [9535, 4386, 2272, 220, 220, 220],
      decoded: "trailing space   ",
    },
    DOUBLE_SPACE: {
      text: BASE_TEST_STRINGS.DOUBLE_SPACE,
      tokens: ["Hi", "\u0120", "\u0120Hello"],
      ids: [17250, 220, 18435],
      decoded: "Hi  Hello",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$", "1", "\u0120R", "2", "\u0120#", "3", "\u0120\u00e2\u0124\u00ac", "4", "\u0120\u00c2\u00a3", "5", "\u0120\u00c2\u00a5", "6", "\u0120\u00e2", "\u0124", "\u00a3", "7", "\u0120\u00e2", "\u0124", "\u00b9", "8", "\u0120\u00e2", "\u0124", "\u00b1", "9", "\u0120test"],
      ids: [9288, 720, 16, 371, 17, 1303, 18, 10432, 19, 4248, 20, 38221, 21, 2343, 224, 96, 22, 2343, 224, 117, 23, 2343, 224, 109, 24, 1332],
      decoded: "test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "\u0120bought", "\u0120an", "\u0120apple", "\u0120for", "\u0120$", "1", ".", "00", "\u0120at", "\u0120the", "\u0120store", "."],
      ids: [40, 5839, 281, 17180, 329, 720, 16, 13, 405, 379, 262, 3650, 13],
      decoded: "I bought an apple for $1.00 at the store.",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u0120", "\u0120"],
      ids: [5832, 1399, 220, 220],
      decoded: "you\u2026  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [5832, 1399, 4603],
      decoded: "you\u2026\u00a0\u00a0",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2\u0142", "\u00c2\u0142", "you", "\u00e2\u0122\u00a6", "\u00c2\u0142\u00c2\u0142"],
      ids: [5832, 1399, 1849, 1849, 5832, 1399, 4603],
      decoded: "you\u2026\u00a0\u00a0you\u2026\u00a0\u00a0",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120\u00ef", "\u00bd", "\u0140", "\u0120edge", "\u0120\u00ef", "\u00bd", "\u0140", "\u0120case"],
      ids: [732, 1447, 27332, 121, 252, 5743, 27332, 121, 252, 1339],
      decoded: "weird \uff5e edge \uff5e case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u00e2\u0138", "\u0123", "This", "\u0120\u00e2\u0138", "\u0123", "is", "\u0120\u00e2\u0138", "\u0123", "a", "\u0120\u00e2\u0138", "\u0123", "test", "\u0120\u00e2\u0138", "\u0123", "."],
      ids: [5008, 223, 1212, 11019, 223, 271, 11019, 223, 64, 11019, 223, 9288, 11019, 223, 13],
      decoded: "\u2581This \u2581is \u2581a \u2581test \u2581.",
    },
    SPECIAL_WITH_TRAILING_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_WITH_TRAILING_WHITESPACE,
      tokens: ["<", "s", ">", "\u010a"],
      ids: [27, 82, 29, 198],
      decoded: "<s>\n",
    },
    SPECIAL_SURROUNDED_BY_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_SURROUNDED_BY_WHITESPACE,
      tokens: ["\u0120</", "s", ">", "\u0120test", "\u0120</", "s", ">", "\u0120"],
      ids: [7359, 82, 29, 1332, 7359, 82, 29, 220],
      decoded: " </s> test </s> ",
    },
    SPECIAL_NO_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_NO_WHITESPACE,
      tokens: ["</", "s", ">", "test", "</", "s", ">"],
      ids: [3556, 82, 29, 9288, 3556, 82, 29],
      decoded: "</s>test</s>",
    },
  },
  // - clean_up_tokenization_spaces=false
  // - custom pretokenization regex
  "Xenova/gpt-4": {
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "\u010a", "'ll", "\u0120!!", "to", "?'", "d", "''", "d", "\u0120of", ",", "\u0120can", "'t", "."],
      ids: [32, 198, 3358, 11261, 998, 20837, 67, 4708, 67, 315, 11, 649, 956, 13],
      decoded: "A\n'll !!to?'d''d of, can't.",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".toString", "();\u010a", "toString", "();"],
      ids: [1169, 264, 284, 2909, 5180, 545, 6712, 2178],
      decoded: "let a = obj.toString();\ntoString();",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$", "1", "\u0120R", "2", "\u0120#", "3", "\u0120\u00e2\u0124\u00ac", "4", "\u0120\u00c2\u00a3", "5", "\u0120\u00c2\u00a5", "6", "\u0120\u00e2", "\u0124", "\u00a3", "7", "\u0120\u00e2\u0124\u00b9", "8", "\u0120\u00e2", "\u0124", "\u00b1", "9", "\u0120test"],
      ids: [1985, 400, 16, 432, 17, 674, 18, 13281, 19, 7083, 20, 72588, 21, 2928, 224, 96, 22, 90891, 23, 2928, 224, 109, 24, 1296],
      decoded: "test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120", "\u00ef\u00bd\u0140", "\u0120edge", "\u0120", "\u00ef\u00bd\u0140", "\u0120case"],
      ids: [906, 2668, 220, 21909, 6964, 220, 21909, 1162],
      decoded: "weird \uff5e edge \uff5e case",
    },
  },
  "Xenova/gpt-4o": {
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["012", "345", "678", "9", "Ġ", "0", "Ġ", "1", "Ġ", "2", "Ġ", "3", "Ġ", "4", "Ġ", "5", "Ġ", "6", "Ġ", "7", "Ġ", "8", "Ġ", "9", "Ġ", "10", "Ġ", "100", "Ġ", "100", "0"],
      ids: [19267, 22901, 30833, 24, 220, 15, 220, 16, 220, 17, 220, 18, 220, 19, 220, 20, 220, 21, 220, 22, 220, 23, 220, 24, 220, 702, 220, 1353, 220, 1353, 15],
      decoded: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "\u0120company", "\u0120was", "\u0120founded", "\u0120in", "\u0120", "201", "6", "."],
      ids: [976, 3175, 673, 24303, 306, 220, 667, 21, 13],
      decoded: "The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "\u010a", "'ll", "\u0120!!", "to", "?'", "d", "''", "d", "\u0120of", ",", "\u0120can't", "."],
      ids: [32, 198, 6090, 17131, 935, 48511, 67, 5830, 67, 328, 11, 8535, 13],
      decoded: "A\n'll !!to?'d''d of, can't.",
    },
    PYTHON_CODE: {
      text: BASE_TEST_STRINGS.PYTHON_CODE,
      tokens: ["def", "\u0120main", "():\u010a", "\u0109pass"],
      ids: [1314, 2758, 8595, 100653],
      decoded: "def main():\n\tpass",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".to", "String", "();\u010a", "to", "String", "();"],
      ids: [1347, 261, 314, 4099, 3552, 916, 740, 935, 916, 4177],
      decoded: "let a = obj.toString();\ntoString();",
    },
    NEWLINES: {
      text: BASE_TEST_STRINGS.NEWLINES,
      tokens: ["This", "\u010a\u010a", "is", "\u010a", "a", "\u010a", "test", "."],
      ids: [2500, 279, 276, 198, 64, 198, 3190, 13],
      decoded: "This\n\nis\na\ntest.",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "want", "\u00c3\u00a9d", ",r", "unning"],
      ids: [2926, 72517, 6383, 33654, 11244],
      decoded: "UNwant\u00e9d,running",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141\u00e6\u00b4\u00bb", "\u00e7\u013c\u0126", "\u00e7\u013e\u0141", "\u00e8\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [32479, 1616, 7910, 7856, 249, 3221],
      decoded: "\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    LEADING_SPACE: {
      text: BASE_TEST_STRINGS.LEADING_SPACE,
      tokens: ["\u0120\u0120", "\u0120leading", "\u0120space"],
      ids: [256, 8117, 4918],
      decoded: "   leading space",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["tr", "ailing", "\u0120space", "\u0120\u0120\u0120"],
      ids: [371, 24408, 4918, 271],
      decoded: "trailing space   ",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$", "1", "\u0120R", "2", "\u0120#", "3", "\u0120\u00e2\u0124\u00ac", "4", "\u0120\u00c2\u00a3", "5", "\u0120\u00c2\u00a5", "6", "\u0120\u00e2\u0124", "\u00a3", "7", "\u0120\u00e2\u0124\u00b9", "8", "\u0120\u00e2\u0124", "\u00b1", "9", "\u0120test"],
      ids: [3190, 548, 16, 460, 17, 1069, 18, 7950, 19, 8989, 20, 123814, 21, 59790, 96, 22, 73406, 23, 59790, 109, 24, 1746],
      decoded: "test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u0120\u0120"],
      ids: [13320, 1131, 256],
      decoded: "you\u2026  ",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120\u00ef\u00bd\u0140", "\u0120edge", "\u0120\u00ef\u00bd\u0140", "\u0120case"],
      ids: [854, 2716, 105665, 11165, 105665, 1890],
      decoded: "weird \uff5e edge \uff5e case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u00e2\u0138", "\u0123", "This", "\u0120\u00e2\u0138\u0123", "is", "\u0120\u00e2\u0138\u0123", "a", "\u0120\u00e2\u0138\u0123", "test", "\u0120\u00e2\u0138\u0123", "."],
      ids: [6762, 223, 2500, 39960, 276, 39960, 64, 39960, 3190, 39960, 13],
      decoded: "\u2581This \u2581is \u2581a \u2581test \u2581.",
    },
    SPECIAL_WITH_TRAILING_WHITESPACE: {
      text: SENTENCEPIECE_TEST_STRINGS.SPECIAL_WITH_TRAILING_WHITESPACE,
      tokens: ["<s", ">\u010a"],
      ids: [101950, 523],
      decoded: "<s>\n",
    },
  },
  "Xenova/claude-tokenizer": {
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".", "toString", "();", "\u010a", "toString", "();"],
      ids: [1785, 269, 284, 2652, 18, 26492, 4370, 203, 26492, 4370],
      decoded: "let a = obj.toString();\ntoString();",
    },
    BASIC: {
      text: BASE_TEST_STRINGS.BASIC,
      tokens: ["UN", "want", "\u00c3\u00a9d", ",", "running"],
      ids: [2359, 17571, 37911, 16, 7889],
      decoded: "UNwant\u00e9d,running",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141", "\u00e6\u00b4\u00bb", "\u00e7\u013c\u0126", "\u00e7\u013e\u0141", "\u00e8\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [14706, 37675, 2471, 56904, 15959, 254, 5977],
      decoded: "\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["trailing", "\u0120space", "\u0120\u0120\u0120"],
      ids: [40110, 3384, 264],
      decoded: "trailing space   ",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$", "1", "\u0120R", "2", "\u0120#", "3", "\u0120\u00e2\u0124\u00ac", "4", "\u0120\u00c2\u00a3", "5", "\u0120\u00c2", "\u00a5", "6", "\u0120\u00e2", "\u0124", "\u00a3", "7", "\u0120\u00e2", "\u0124", "\u00b9", "8", "\u0120\u00e2", "\u0124", "\u00b1", "9", "\u0120test"],
      ids: [765, 734, 21, 487, 22, 379, 23, 36714, 24, 13206, 25, 2455, 103, 26, 4937, 229, 101, 27, 4937, 229, 122, 28, 4937, 229, 114, 29, 722],
      decoded: "test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    ELLIPSIS: {
      text: BASE_TEST_STRINGS.ELLIPSIS,
      tokens: ["you", "...", "\u0120\u0120"],
      ids: [6773, 1174, 261],
      decoded: "you...  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "...", "\u0120\u0120"],
      ids: [6773, 1174, 261],
      decoded: "you...  ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "...", "\u0120", "\u0120you", "...", "\u0120\u0120"],
      ids: [6773, 1174, 225, 583, 1174, 261],
      decoded: "you...  you...  ",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120~", "\u0120edge", "\u0120~", "\u0120case"],
      ids: [798, 2650, 6217, 4915, 6217, 1544],
      decoded: "weird ~ edge ~ case",
    },
  },
  "bigcode/santacoder": {
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Ġ", "0", "Ġ", "1", "Ġ", "2", "Ġ", "3", "Ġ", "4", "Ġ", "5", "Ġ", "6", "Ġ", "7", "Ġ", "8", "Ġ", "9", "Ġ", "1", "0", "Ġ", "1", "0", "0", "Ġ", "1", "0", "0", "0"],
      ids: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 207, 15, 207, 16, 207, 17, 207, 18, 207, 19, 207, 20, 207, 21, 207, 22, 207, 23, 207, 24, 207, 16, 15, 207, 16, 15, 15, 207, 16, 15, 15, 15],
      decoded: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "\u0120company", "\u0120was", "\u0120fo", "unded", "\u0120in", "\u0120", "2", "0", "1", "6", "."],
      ids: [2111, 10107, 2501, 17436, 7584, 319, 207, 17, 15, 16, 21, 13],
      decoded: "The company was founded in 2016.",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141", "\u00e6\u00b4\u00bb", "\u00e7\u013c\u0126", "\u00e7\u013e\u0141", "\u00e8\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [8715, 24543, 1825, 34717, 37452, 236, 4343],
      decoded: "\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "\u0120bo", "ught", "\u0120an", "\u0120apple", "\u0120for", "\u0120$", "1", ".", "0", "0", "\u0120at", "\u0120the", "\u0120store", "."],
      ids: [40, 12307, 10310, 743, 29806, 408, 763, 16, 13, 15, 15, 869, 331, 2823, 13],
      decoded: "I bought an apple for $1.00 at the store.",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120", "\u00ef\u00bd", "\u0140", "\u0120edge", "\u0120", "\u00ef\u00bd", "\u0140", "\u0120case"],
      ids: [1850, 4427, 207, 29217, 239, 4959, 207, 29217, 239, 1210],
      decoded: "weird \uff5e edge \uff5e case",
    },
    SPIECE_UNDERSCORE: {
      text: BASE_TEST_STRINGS.SPIECE_UNDERSCORE,
      tokens: ["\u00e2\u0138", "\u0123", "This", "\u0120", "\u00e2\u0138", "\u0123", "is", "\u0120", "\u00e2\u0138", "\u0123", "a", "\u0120", "\u00e2\u0138", "\u0123", "test", "\u0120", "\u00e2\u0138", "\u0123", "."],
      ids: [3718, 210, 3456, 207, 3718, 210, 280, 207, 3718, 210, 64, 207, 3718, 210, 706, 207, 3718, 210, 13],
      decoded: "\u2581This \u2581is \u2581a \u2581test \u2581.",
    },
  },
  "Xenova/CodeGPT-tokenizer": {
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141", "\u00e6", "\u00b4", "\u00bb", "\u00e7\u013c\u0126", "\u00e7\u013e", "\u0141", "\u00e8\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [25506, 165, 115, 122, 5137, 43415, 256, 20679, 252, 13283],
      decoded: "\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["trailing", "\u0120space", "\u0120", "\u0120", "\u0120"],
      ids: [15584, 3497, 223, 223, 223],
      decoded: "trailing space   ",
    },
    TEXT_WITH_ESCAPE_CHARACTERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2", "\u0142", "\u00c2", "\u0142"],
      ids: [13953, 29502, 129, 257, 129, 257],
      decoded: "you\u2026\u00a0\u00a0",
    },
    TEXT_WITH_ESCAPE_CHARACTERS_2: {
      text: BASE_TEST_STRINGS.TEXT_WITH_ESCAPE_CHARACTERS_2,
      tokens: ["you", "\u00e2\u0122\u00a6", "\u00c2", "\u0142", "\u00c2", "\u0142", "you", "\u00e2\u0122\u00a6", "\u00c2", "\u0142", "\u00c2", "\u0142"],
      ids: [13953, 29502, 129, 257, 129, 257, 13953, 29502, 129, 257, 129, 257],
      decoded: "you\u2026\u00a0\u00a0you\u2026\u00a0\u00a0",
    },
  },
  "huggingface-course/codeparrot-ds": {
    NUMBERS: {
      text: BASE_TEST_STRINGS.NUMBERS,
      tokens: ["0123456789", "Ġ0", "Ġ1", "Ġ2", "Ġ3", "Ġ4", "Ġ5", "Ġ6", "Ġ7", "Ġ8", "Ġ9", "Ġ10", "Ġ100", "Ġ1000"],
      ids: [25218, 443, 396, 554, 869, 1163, 1462, 1911, 2624, 2070, 2837, 2009, 3038, 4764],
      decoded: "0123456789 0 1 2 3 4 5 6 7 8 9 10 100 1000",
    },
    TEXT_WITH_NUMBERS: {
      text: BASE_TEST_STRINGS.TEXT_WITH_NUMBERS,
      tokens: ["The", "\u0120company", "\u0120was", "\u0120fo", "unded", "\u0120in", "\u01202016", "."],
      ids: [2096, 16502, 1442, 11689, 7865, 253, 8780, 14],
      decoded: "The company was founded in 2016.",
    },
    PUNCTUATION: {
      text: BASE_TEST_STRINGS.PUNCTUATION,
      tokens: ["A", "\u010a", "'ll", "\u0120!", "!", "to", "?'", "d", "''", "d", "\u0120of", ",", "\u0120can", "'t", "."],
      ids: [33, 173, 6402, 905, 1, 403, 15227, 68, 589, 68, 311, 12, 796, 1059, 14],
      decoded: "A\n'll!!to?'d''d of, can't.",
    },
    JAVASCRIPT_CODE: {
      text: BASE_TEST_STRINGS.JAVASCRIPT_CODE,
      tokens: ["let", "\u0120a", "\u0120=", "\u0120obj", ".", "toString", "();", "\u010a", "toString", "();"],
      ids: [2047, 231, 233, 1300, 14, 30494, 16248, 173, 30494, 16248],
      decoded: "let a = obj.toString();\ntoString();",
    },
    CHINESE_ONLY: {
      text: BASE_TEST_STRINGS.CHINESE_ONLY,
      tokens: ["\u00e7\u0136\u0141", "\u00e6\u00b4", "\u00bb", "\u00e7\u013c\u0126", "\u00e7\u013e", "\u0141", "\u00e8\u00b0", "\u013d", "\u00e6\u013a\u00af"],
      ids: [20185, 43799, 120, 3994, 37782, 211, 15933, 207, 11130],
      decoded: "\u751f\u6d3b\u7684\u771f\u8c1b\u662f",
    },
    TRAILING_SPACE: {
      text: BASE_TEST_STRINGS.TRAILING_SPACE,
      tokens: ["trailing", "\u0120space", "\u0120\u0120\u0120"],
      ids: [17031, 3000, 216],
      decoded: "trailing space   ",
    },
    CURRENCY: {
      text: BASE_TEST_STRINGS.CURRENCY,
      tokens: ["test", "\u0120$", "1", "\u0120R", "2", "\u0120#", "3", "\u0120\u00e2", "\u0124\u00ac", "4", "\u0120\u00c2", "\u00a3", "5", "\u0120\u00c2", "\u00a5", "6", "\u0120\u00e2", "\u0124", "\u00a3", "7", "\u0120\u00e2", "\u0124", "\u00b9", "8", "\u0120\u00e2", "\u0124", "\u00b1", "9", "\u0120test"],
      ids: [1824, 3748, 17, 683, 18, 294, 19, 5161, 28898, 20, 23446, 97, 21, 23446, 99, 22, 5161, 182, 97, 23, 5161, 182, 118, 24, 5161, 182, 110, 25, 1737],
      decoded: "test $1 R2 #3 \u20ac4 \u00a35 \u00a56 \u20a37 \u20b98 \u20b19 test",
    },
    CURRENCY_WITH_DECIMALS: {
      text: BASE_TEST_STRINGS.CURRENCY_WITH_DECIMALS,
      tokens: ["I", "\u0120bo", "ught", "\u0120an", "\u0120app", "le", "\u0120for", "\u0120$", "1", ".", "00", "\u0120at", "\u0120the", "\u0120store", "."],
      ids: [41, 772, 8272, 309, 870, 239, 296, 3748, 17, 14, 543, 815, 256, 2689, 14],
      decoded: "I bought an apple for $1.00 at the store.",
    },
    TILDE_NORMALIZATION: {
      text: BASE_TEST_STRINGS.TILDE_NORMALIZATION,
      tokens: ["we", "ird", "\u0120", "\u00ef", "\u00bd", "\u0140", "\u0120edge", "\u0120", "\u00ef", "\u00bd", "\u0140", "\u0120case"],
      ids: [955, 6075, 179, 166, 122, 210, 2703, 179, 166, 122, 210, 1539],
      decoded: "weird \uff5e edge \uff5e case",
    },
  },
};

// Test that tokenizer type can be inferred (`type: "BPE"` is missing)
TEST_CONFIG["openai-community/gpt2"] = TEST_CONFIG["Xenova/gpt2"];
