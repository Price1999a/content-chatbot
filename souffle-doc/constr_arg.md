---
title: Constraints and Arguments
permalink: /constr_arg
sidebar: docs_sidebar
folder: docs
---

## Strings

* **cat(*string*, *string*)** is used to concatenate two strings together. It can be nested to concatenate more than two strings.
```prolog
.decl Y(a:symbol, b:symbol)
.decl Z(a:symbol, b:symbol, c:symbol)
.output Z
Y("a","b").
Y("c","d").
Z(a,b, cat(cat(a,b), a)) :- Y(a,b).
```
The output would be:
```
a	b	aba
c	d	cdc
```

* **contains(*string1*, *string2*)** is used to check if the latter string contains the former string.
```prolog
.decl stringTable(t:symbol)
.decl substringTable(t:symbol)
.decl outputData(substr:symbol, str:symbol)
.output outputData
outputData(x,y) :- substringTable(x), stringTable(y), contains(x,y).
stringTable("aaaa").
stringTable("abba").
stringTable("bcab").
stringTable("bdab").
substringTable("a").
substringTable("ab").
substringTable("cab").
```
The output would be:
```
a	aaaa
a	abba
a	bcab
a	bdab
ab	abba
ab	bcab
ab	bdab
cab	bcab
```

* **match** is used to check if the latter string matches a wildcard pattern specified in the former string.
```prolog
.decl inputData(t:symbol)
.decl outputData(t:symbol)
.output outputData
outputData(x) :- inputData(x), match("a.*",x).
inputData("aaaa").
inputData("abba").
inputData("bcab").
inputData("bdab").
```
The output would be:
```
aaaa
abba
```

* **ord(*string*)** is used to return the ordinal number associated with *string*. **This is not a lexicographic ordering.** The ordinal number is based on the order of appearance (see example below).
```prolog
.decl n(x:symbol)
n("Homer").
n("Marge").
n("Bart").
n("Lisa").
n("Maggie").
.decl r(x:number)
.output r
r(1) :- n(x), n(y), ord(x) < ord(y), x="Homer", y="Bart".
r(2) :- n(x), n(y), ord(x) > ord(y), x="Maggie", y="Homer".
r(3) :- n(x), n(y), ord(x) > ord(y), x="Marge", y="Bart".
```
The output would be:
```
1
2
```
`r(3)` is not set, since ord("Marge") is less than ord("Bart") (the string "Marge" appears before the string "Bart", therefore it has a smaller ordinal number).

* Equality operations (**=** and **!=**) are also available for string types, by performing an ordinal comparison.

* **strlen(*string*)** returns the length of *string* as number.
```prolog
.decl length(n:number)
.output length
length(n) :- n=strlen("Hello").
length(n) :- n=strlen("World!").
```
The output would be:
```
5
6
```

* **substr(*string*, *index*, *length*)** is used to return the substring starting at *index* with length *length* of *string*. The index is zero-based.
```prolog
.decl substring(s:symbol)
.output substring
substring(s) :- s=substr("Hello_", 2, 3).
substring(s) :- string="World!", s=substr(string, 3, strlen(string)).
```
The output would be:
```
llo
ld!
```

* **to_number(*string*)** transforms a string representing a number to its associated number.
```prolog
.decl tonumber(n:number)
.output tonumber
tonumber(n) :- n=to_number("123").
tonumber(n) :- n=to_number("1534").
```
The output would be:
```
123
1534
```
The reverse operation **to_string(*number*)** also exists, which turns a number to its string representation.

## Arithmetic 

* Soufflé supports standard arithmetic operations **+**, **-**, **&#42;**, **&#47;**, **&#94;** and **&#37;**. Examples of this are given below.
```
.decl e(x:number, t:symbol, y:number)
e(10 * 2,"10*2", 20).
e(10 + 2,"10+2", 12).
e(10 / 2,"10/2", 5).
e(10 ^ 2 , "10^2", 100).
e(10 % 3, "10%3", 1).
e(2^4%13 , "2^4%13",3).
```
* Soufflé supports bitwise logical operations: **band** (bitwise and), **bor** (bitwise or), **bxor** (bitwise exclusive-or), **bshl** (bitwise shift left), **bshr** (bitwise shift right), and **bshru** (bitwise shift right/unsigned). 

Examples of this are given below.
```
e(0xFFF1 band 0xF, "0xFFF1 band 0xF", 0x1).
e(0xFF00 bor 0x000F, "0xFF00 bor 0x000F", 0xFF0F).
e(0xFFFF bxor 0x000F, "0xFFFF bxor 0x000F", 0xFFF0).
```

* Soufflé supports logical operations that consider every non-zero number as true and always return 1 or 0: **land** (logical and), **lor** (logical or), **lxor** (logical exclusive-or), and **lnot** (logical not).

Examples of this are given below.
```
e(1 land 2, "1 land 2", 1).
e(1 land 0, "1 land 0", 0).
e(1 lor 0, "1 lor 0", 1).
```

* Soufflé supports **max** and **min** operations over numbers.
```
e(max(3, 4), "max(3, 4)", 4).
e(min(3, 4), "min(3, 4)", 3).
```

* Soufflé supports standard unary operation **-**.
```
e(-2*10,"-20", -20).
e(-2,"-2", -2).
e(--2,"--2", 2).
```

* Soufflé supports standard binary operations **&#62;**, **&#60;**, **&#61;**, **&#33;&#61;**, **&#62;&#61;** and **&#60;&#61;**. Examples of this are given below.
```
A(a,c) :- a > c.
B(a,c) :- a < c.
C(a,c) :- a = c.
D(a,c) :- a != c.
E(a,c) :- a <= c.
F(a,c) :- a >= c.
```

* **&#36;** is used to generate unique random values to populate a table. It should be used with care as it may result in stepping outside the standard Datalog semantics.
```
.decl A (n:number)
.decl B (a:number, b:number)
.decl C (a:number, b:number)
.output C
A(0).
A(i+1) :- A(i), i<1000.		
B($,i) :- A(i).		
C(i,j) :- B(c,i), B(c,j), i!=j.
```
The above example does not output anything.

## Syntax 
In the following, we define constraints and argument values in Soufflé more formally using [syntax diagrams](https://en.wikipedia.org/wiki/Syntax_diagram) and [EBNF](https://en.wikipedia.org/wiki/Extended_Backus–Naur_form). The syntax diagrams were produced with [Bottlecaps](https://www.bottlecaps.de/rr/ui).

### Constraints

![constraint](https://souffle-lang.github.io/img/constraint.svg)

```ebnf
constraint ::= argument ( '<' | '>' | '<=' | '>=' | '=' | '!=' ) argument
           | ( 'match' | 'contains' ) '(' argument ',' argument ')'
           | 'true'
           | 'false'
```


### Argument List

![Argument List](https://souffle-lang.github.io/img/argument_list.svg)

```ebnf
argument_list ::= ( argument ( ',' argument )* )?
```

### Argument Values

![Argument](https://souffle-lang.github.io/img/argument.svg)

```ebnf
argument ::= 
      STRING
    | FLOAT
    | UNSIGNED
    | NUMBER
    | '_'
    | '$' ( IDENT ( '(' argument_list ')' )? )?
    | IDENT
    | 'nil'
    | '[' argument_list ']'
    | ( '(' argument | 'as' '(' argument ',' type_name | ( '@' IDENT | intrinsic_functor ) '(' argument_list ) ')'
    | aggregator
    | ( unary_operation | argument binary_operation ) argument     
```


### Unary Operations

![Unary Operation](https://souffle-lang.github.io/img/unary_operation.svg)

```ebnf
unary_operation ::= '-' | 'bnot' | 'lnot'
```

### Binary Operations

![Binary Operation](https://souffle-lang.github.io/img/binary_operation.svg)

```ebnf
binary_operation ::=  '+' | '-' | '*' | '/' | '%' | '^' | 'land' | 'lor' | 'lxor' | 'band' | 'bor' | 'bxor' | 'bshl' | 'bshr' | 'bshru' 
```

### Intrinsic Functors

![Intrinsic Functor](https://souffle-lang.github.io/img/intrinsic_functor.svg)

```ebnf
intrinsic_functor ::= 'cat' | 'ord' | 'range' | 'strlen' | 'substr' | 'to_float' | 'to_number' | 'to_string' | 'to_unsigned'
```           

{% include links.html %}
