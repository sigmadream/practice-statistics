# Introduction

## The Research Assistant

You are back in DC! Yes. It is as hot and humid as it was for your internship, maybe even hotter and humiditier. You managed to secure a job as an “RA” at the US Federal Trade Commission. You are in the Bureau of Economics in AT1\. You have an awesome place in Navy Yard, not far from the Nats stadium, the Wharf and the home for DC United. A mate from college is your room mate and they started at the Fed last week. Your career is about to begin!

### First Day

What the heck is he talking about? A colleague, was it John or Jeff, maybe Dave, has been talking 100 mph for last few minutes and you are just catching snippets. “We need to estimate WTP.” “Can you believe they are using Elzinga-Hogarty.” “I hope Alabama joins the suit.” “Bill Town is great.” You have been nodding but you quickly lost the thread of the conversation.

You decided to go with a suit, which was definitely a mistake. Even though you were hardly outside, you are sweating buckets and no one else seems to be in a suit, not even the managers. The managers are lovely and seemed very excited that you were starting. They liked that you had coding experience in **R** and had done some cool work in your internship on minimum wages. Apparently, you are going to jump straight into a case.

John/Jeff/maybe-Dave is explaining the case to you. It seems to have something to do with hospitals in Alabama. Your screen is full of data, there are columns with things you recognize like zip codes, age, and then things like DRG which you have no idea about. Apparently, this data is from a payer, although you are not sure what that is. The rows are claims. There is a column with the price, but they seem wrong. The numbers are enormous, 150008, 25020, 83251\. Is this dollars? You were in hospital a few weeks back getting some stitches after an incident playing kick ball. You are pretty sure it was $25.00

Your job is to determine the relationship between price and WTP. You just wish you knew what WTP stood for.

### Second Day

Yesterday you were able to get **R** and RStudio set up and get access to the payer data. You were able to calculate WTP for the hospitals and you showed Dave the results.

OK. Now we need to calculate WTP post-merger. You responded that you didn't think you had the data, you were pretty sure that there was no indicator for merger in the data. Dave laughed out loud. Then he saw the expression on your face, caught himself, and stated matter-of-factly, no we have to simulate the merger.

Simulate the merger? How would we know what would happen to prices after the hospitals merge? How do we know how prices are determined now? From what you could gather so far there were three hospitals in the city and five insurance companies serving beneficiaries in the area. The prices in the data were determined by bargaining between these hospitals and the insurance companies. You search your memory back to microeconomics, you remember the class on monopoly pricing, when the seller had market power to determine price. Do these hospitals have market power? You remember something in the text book about monopsony, when the buyer has market power to determine price. Do the insurers have market power? Do both the hospitals and the insurers have market power?

## The Book

This is an empirical game theory book. Traditionally, game theory is presented as a theoretical subject. Generally, applications are discussed but there are no explicit empirical applications. Yet, since the 1990s, game theory has been at the heart of empirical analysis of competition and markets in the economics sub-field of industrial organization. In economics, this layering of theory on to empirical analysis is called **structural econometrics**. This book is focused on the game theory, not necessarily the econometrics.

### What Does it Cover?

The book aims to provide an introduction game theory, a mathematical approach to understanding economic relationships. The goal is for the reader to understand what a game is and how the mathematics works. The reader will be able to create a game that explains behavior of economic actors observed in her data. She will be able to ask _how_ questions. She will be able to see how parameters of the game relate to characteristics of the data. She may be able to use the game to ask _what if_ questions. What would the economic actors observed in the data do if the world was different? What if the government introduced a new policy? What if technology changed. What if the hospitals merged?

The book covers standard game theory concepts such as **normal form games, exentensive form games, Nash equilibrium, mixed strategy Nash equilibrium**, and **subgame perfection**. It also covers standard empirical methods such as **linear regression, two-way fixed effects, logit models**, and **maximum likelihood estimation**. But it covers a number of concepts that are important to the intersection of game theory and data analysis such as **generalized method of moments** and **two-step estimators**.

### What is the Approach?

The book teaches game theory through code, in particular, it will use the scripting language **R**. It is not primarily aimed at teaching **R**. Rather, it is primarily aimed at teaching game theory. This idea of using computer programming as a tool of instruction goes back to at least Seymour Papert and MIT's AI lab in the 1970s.[1](./07-fm-chapter0.md#fnfm_1) Papert helped develop a programming language called Logo. The goal of Logo was to teach geometry by programming how a **turtle** moves around the screen. You may have used one of the offspring of Logo, such as Scratch or Lego Mindstorms.

The book uses Papert's ideas to teach game theory. You will learn the math of the game or estimation method and then how to program that game or estimation method. The book makes particular use of the computer's ability to simulate data. This allows us to experiment with more complicated and realistic games than is possible with pen and paper.

The book is written in RStudio using Sweave. Sweave allows LaTeX to be integrated into **R**. LaTeX is a free type-setting language that is designed for writing math. Much of the code that is used in the book is actually presented in the book. Sometimes it is more practical to create a data set outside the of book. In those cases, the data and the code that created the data are available here <https://github.com/christopherpadams/EmpiricalGameTheory>. In a couple of other cases, the preferred code does not produce nice output for the book, so it is left out.

### What is How Analysis?

Often we want to understand patterns we see in the data. How are prices related to the number of firms in the market? Why do firms enter some markets and not others?

For those questions, we can use game theory to guide our thinking and our analysis. A game theoretic model of firm entry can help us understand why some markets had both Borders and Barnes & Noble, other markets had just one, and many more have none.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [1](./07-fm-chapter0.md#fnfm_1b)<https://el.media.mit.edu/logo-foundation/>

### What is What If Analysis?

Game theory is also important in _what if_ analysis. In those analysis, we want to understand how behavior will change when faced with a situation not observed in the data. What would have happened if Borders and Barnes & Noble would have merged in the mid-1990s? Would there have been more or fewer mega bookstores? Should the federal government allow bidders to collude in oil drilling auctions? We can estimate the parameters of game theoretic model using observed data and then make changes to the model to simulate what would happen in the case that is not observed in the data. We can then run the model and predict the outcome. We can simulate a merger between Borders and Barnes & Noble in the mid-1990s or what happens to bids on oil drilling leases when bidders are allowed to collude.

### What About the Real World?

The book presents interesting and important questions. The book presents an analysis of competition between various types of firms and asks what happens if mergers are allowed or not. It considers price regulation policies for retail gasoline and whether these regulations lead to higher prices for consumers. It looks at how the US federal government runs auctions for timber and oil drilling leases. Hopefully, the book points you to new questions and new data to answer existing questions.

The book does not recommend policies. The government economist, Alice Rivlin, argued that it is extremely important to provide policy makers with objective analysis. In a memo to staff of the Congressional Budget Office (CBO), she said the following.[2](./07-fm-chapter0.md#fnfm_2)

> We are not to be advocates. As private citizens, we are entitled to our own views on the issues of the day, but as members of CBO, we are not to make recommendations, or characterize, even by implication, particular policy questions as good or bad, wise or unwise.

Economists in government, the private sector and the academy, work on important policy questions. Economists are most effective when they do not advocate for policy positions, but present objective analysis of the economics and the data. This book presents an objective analysis of interesting policy questions but doesn't state whether the policy positions are good or bad, wise or unwise.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [2](./07-fm-chapter0.md#fnfm_2b)<https://www.cbo.gov/sites/default/files/Public%5FPolicy%5FIssues%5FMemo%5FRivlin%5F1976.pdf>

## The Outline

The book is laid out the same way Robert Gibbons laid out his classic text, _Game Theory for Applied Economists_. There are four parts. **Static games of complete information**, dynamic games of complete information, static games of incomplete information, and dynamic games of incomplete information.

### Static Games of Complete Information

This part presents the simplest version of a game.

[Chapter 1](./09-chapter1.md) introduces the basic mathematical concepts of game theory. It analyzes the most famous game in game theory, the **prisoner's dilemma**. The game is used in a TV game show and the chapter uses data on actual behavior in the game where the players of the game make choices worth thousands of dollars. Do real people on a TV game show play the game as the mathematics predicts?

[Chapter 2](./10-chapter2.md) introduces two important equilibrium concepts, **dominant strategy equilibrium** and **Nash equilibrium**. The chapter uses Nash equilibrium to understand how the number of tire retailers varies from city to city.

[Chapter 3](./11-chapter3.md) studies **oligopoly**, markets with a small number of competitors, and three models of how these markets work, **Cournot, Bertrand**, and **Hotelling**. The most general model allows competing firms to be similar but not the same. This model is used to understand pricing competition between McDonald's outlets in late 1990s Santa Clara county.

[Chapter 4](./12-chapter4.md) considers the implications of multiple Nash equilibria in a game where two firms are choosing whether or not to enter the same market. This game is used to analyze the entry decisions by the mega bookstores in the 1990s, Borders and Barnes & Noble. The chapter analyzes problems where the game does not always make a single prediction.

[Chapter 5](./13-chapter5.md) analyzes **mixed strategies** in the context of both **coordination games** and **zero-sum games**. It uses mixed strategies to model entry by the mega bookstores, Borders, and Barnes & Noble. Zero-sum games were the first types of games analyzed using game theory. Many parlor games like chess, droughts, and poker are zero-sum games. The chapter uses mixed strategies and zero-sum games to understand the choices made by soccer players when kicking and defending penalty kicks in the English Premier League.

### Dynamic Games of Complete Information

These games are substantially more complicated than the games presented in the first part of the book. In response, we need to make a number of simplifying assumptions that allow us to use the richness of the dynamics without being overwhelmed by the complexities.

[Chapter 6](./15-chapter6.md) introduces **subgame perfection** and uses the concept to analyze the entry dynamics of the mega bookstores, Borders and Barnes & Noble.

[Chapter 7](./16-chapter7.md) presents three different models of **bargaining**. It asks whether the simplest, the **ultimatum game**, makes predictions that are consistent with actual behavior of actual people when making decisions involving large sums of money. The answer is no, not really. A more complicated game makes more reasonable predictions. Luckily the **Rubenstein alternating offers game** makes predictions similar to a much simpler analytical tool, the **Nash bargaining solution**. This tool is used to analyze mergers between hospitals in Palm Beach County, the home county for the publisher of this book.

[Chapter 8](./17-chapter8.md) returns to **oligopoly markets** but allows more complicated interactions between the firms. The chapter analyzes the pricing behavior of gasoline retailers in Perth Australia. The chapter presents a model to explain the weird saw tooth pattern in retail gas prices. It considers the extent of the prediction error when a merger model assumes firms choose prices more independently than they actually do.

### Static Games of Incomplete Information

In the first two parts of the book, the players of the game are assumed to know everything. In the second two parts of the book, that assumption is relaxed.

[Chapter 9](./19-chapter9.md) revisits analysis of entry by mega bookstores, Borders and Barnes & Noble. The difference is that each firm observes its own entry costs but not the entry costs of the other firm. In this model equilibrium, the firms do not know their competitors unobserved costs of entry. The firms know their own costs of entering a new market, but not their competitors costs of entering that same market. While the game is more complicated than the game presented in [Chapter 4](./12-chapter4.md), it is sometimes simpler to use in data analysis.

[Chapter 10](./20-chapter10.md) and [Chapter 11](./21-chapter11.md) analyze auctions. [Chapter 10](./20-chapter10.md) uses two of the main auction types, **sealed bid auctions** and **English auctions** to analyze bidding behavior and collusion in US Forestry auctions conducted in the 1970s.

[Chapter 11](./21-chapter11.md) considers auctions where the bidders don't know exactly how much to value the item they are bidding on. The classic example is oil drilling auctions on the US Outer Continental Shelf (OCS). The chapter asks whether the government should allow firms to collude in those auctions.

### Dynamic Games of Incomplete Information

The fourth part of the book analyzes the problems of moral hazard and adverse selection.

[Chapter 12](./23-chapter12.md) considers the **principal-agent problem** and uses it to analyze the use of corporate financing in 19th-Century whaling in New England. What sort of contracts were used by firms and families to finance whaling expeditions, where the whaling ships were literally sailing around the world?

[Chapter 13](./24-chapter13.md) brings game theory to health insurance markets. The chapter presents a model of the used car market, which suggests that information problems will cause the market to fail. Similarly, information problems in health insurance markets suggest that government interventions such as large tax subsidies for people insured through their work are necessary in order to have people insured.

### Notation

As you have seen above, the book uses particular fonts and symbols for various important things. It uses the symbol **R** to refer to the scripting language. It uses typewriter font to represent code in **R**. Initial mentions of an important term are in **bold face font**.

When discussing actual data, it uses _xi_ to refer to the observed characteristic for some individual _i_. It uses x to denote a vector of the _xi_'s. For matrices, it uses X for a matrix and X′ for the matrix transpose. A row of that matrix is Xi or Xi′ to highlight that it is a row vector. Lastly, for parameters of interest it uses Greek letters. For example, _β_ generally refers to a vector of parameters, although in some cases it is a single parameter of interest, while β^ refers to the estimate of the parameter.

## Hello **R** World

To use this book you need to download **R** and RStudio on your computer. Both are free.

### Download **R** and RStudio

First, download the appropriate version of RStudio here: <https://www.rstudio.com/products/rstudio/download/#download>. Then you can download the appropriate version of **R** here: <https://cran.rstudio.com/>.

Once you have the two programs downloaded and installed, open up RStudio. To open up a _script_ go to “File > New File > R Script.” You should have 4 windows, a script window, a console window, a global environment window, and a window with help, plots, and other things.

### Using the Console

Go to the console window and click on the \>. Then type print(“Hello R World”) and hit enter. Remember to use the quotes. In general, **R** functions have the same basic syntax, functionname with parentheses, and some input inside the parentheses. Inputs in quotes are treated as text while inputs without quotes are treated as variables.


`_> print(“Hello R World”)_`
`  [1] “Hello R World”`
Try something a little more complicated.


`> _a = “Chris” # or write your own name_`
`> _print(paste(“Welcome”,a,“to R World”))_`
`  [1] “Welcome Chris to R World”`
Here we are creating a variable called a. The # is used in **R** to “comment out” lines in codes. **R** does not read the line following the hash.

In **R** we can place one function inside another function. The function paste is used to join text and variables together. The function paste() defaults to placing a space between the inputs. When placing one function inside another make sure to keep track of all of the parentheses. A common error is to have more or less closing parentheses than opening parentheses.


`> _paste(_`
`  “_Welcome_”,`
`   _a_,`
`   “_to R World_”`
`) |>`
`  _print(_`
`  _)_`
`  [1] “Welcome Chris to R World”`
**R** can also accept code that looks like above. Using spaces and new lines helps a human reader understand what the code is saying. The symbol | > says take the result from above and use it in the following function. For some reason, there are two different symbols used % > % or | >. You can select which type under global options. Here | > is used.

## Thanks

I'm grateful to my wife, Deena Ackerman, for allowing me time to work on this project. Thank you to numerous friends and colleagues for providing feedback and suggestions on the book. I am particularly grateful to Devesh Raval and Emek Basker who gave extensive feedback on early drafts. Thanks also to a number of researchers who have been willing to provide me with the data used in the book including Raph Thomadsen, Eric Hilt and David Byrne. All errors are my own.

## Discussion and Further Reading

The book is laid out the same way as [Gibbons (1992)](./25-refbib.md#ref27). If you are looking for a more detailed or technical description of the various types of games then see [Fudenberg and Tirole (1991)](./25-refbib.md#ref24). A lot of the applications presented in this book are from the sub-field of economics called industrial organization. The classic theory text for that field is [Tirole (1988)](./25-refbib.md#ref57). Recently more empirical-oriented industrial organization books have come out including [Aguirregabiria (2021)](./25-refbib.md#ref3) and [Hortaçsu and Joo (2023)](./25-refbib.md#ref36). [Paarsch and Hong (2006)](./25-refbib.md#ref49) have a similar orientation with a focus on auctions. If you are interesting in some of the economic experiments presented in the book, then [Camerer (2003)](./25-refbib.md#ref19) is a good but somewhat dated overview of experimental game theory.

The book uses the coding language **R** to illustrate models and empirical problems. It generally uses the tidyverse and data.table flavors. The best introduction to the language is the books by Hadley Wickham and coauthors, in particular _R for Data Science_ which is here <https://r4ds.had.co.nz/>.

[_OceanofPDF.com_](./https___oceanofpdf.com)
